# main.py
import io
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── 한글 폰트 CSS ──────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

st.set_page_config(page_title="새싹팀 길이마스터", layout="wide")


# ── CSV 읽기 (인코딩 이중 시도) ────────────────────────────────────────────
@st.cache_data
def read_csv(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


# ── 열 이름 표준화 (KeyError 방지) ────────────────────────────────────────
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)

    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num

    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]

    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed

    return df


# ── CSV 다운로드 헬퍼 (utf-8-sig) ────────────────────────────────────────
def make_download_csv(df):
    return df.to_csv(index=False).encode("utf-8-sig")


# ── 학교명 추출 ───────────────────────────────────────────────────────────
def extract_school(filename):
    if filename is None:
        return "우리 학교"
    base = os.path.basename(filename)
    name = base.split("_")[0]
    return name if name else "우리 학교"


# ═══════════════════════════════════════════════════════════════════════════
# 로딩 헬퍼
# ═══════════════════════════════════════════════════════════════════════════
def load_env_df(file):
    if file is None:
        return None
    with st.spinner("환경 데이터를 읽는 중..."):
        return std_cols(read_csv(file))


def load_growth_df(file):
    if file is None:
        return None
    with st.spinner("생육 데이터를 읽는 중..."):
        return std_cols(read_csv(file))


# ═══════════════════════════════════════════════════════════════════════════
# 사이드바
# ═══════════════════════════════════════════════════════════════════════════
st.sidebar.title("📁 데이터 업로드")

env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 데이터 CSV", type="csv")

growth_items = st.sidebar.multiselect(
    "생육 측정 항목 선택 (기본 모두 선택)",
    ["잎 수", "지상부 길이", "지하부 길이", "생중량", "지상부 질량", "지하부 질량"],
    default=[
        "잎 수", "지상부 길이", "지하부 길이", "생중량", "지상부 질량", "지하부 질량"
    ],
)

school_name = extract_school(
    growth_file.name if growth_file else env_file.name if env_file else None
)

tabs = st.tabs([
    "📖 실험 개요",
    "🌡️ 환경 데이터",
    "📊 생육 결과",
    "🔬 가설 검증",
    "💡 한계 및 개선방향",
])


# ═══════════════════════════════════════════════════════════════════════════
# 탭 1 — 실험 개요
# ═══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    try:
        st.header(f"🌱 {school_name} 극지식물 개체별 생육 차이 분석")

        col_bg, col_obj = st.columns(2)
        with col_bg:
            st.subheader("연구 배경 및 목적")
            st.markdown(
                f"""
**극지식물**은 척박한 환경에서도 생존하는 식물로, 같은 조건에서 자라도 개체마다 생육 차이가 나타납니다.

이 대시보드는 **{school_name}**에서 재배한 극지식물 개체들의 생육 데이터를 분석하여
- 개체별 생중량 분포와 차이 (평균·표준편차·변동계수 CV)
- 잎 수·지상부 길이와 생중량의 관계 (상관계수·산점도)
- 생중량 상위 25% vs 하위 25% 개체 비교
- 지상부 길이와 지하부 길이의 상관관계 및 비율

를 시각적으로 확인하는 도구입니다.

> **통계 용어 간단 설명**
> - **평균**: 모든 값을 더해 개수로 나눈 대표값
> - **표준편차**: 값들이 평균에서 얼마나 퍼져 있는지 나타내는 지표
> - **변동계수(CV)**: 표준편차를 평균으로 나눈 값 → 단위가 다른 집단 간 비교에 사용
> - **상관계수**: 두 변수가 함께 변화하는 정도 (−1~1, 1에 가까울수록 강한 양의 관계)
"""
            )

        with col_obj:
            st.subheader("데이터 요약")
            if growth_file is not None:
                g_df = load_growth_df(growth_file)
                n_ind = (
                    len(g_df.drop_duplicates(subset=["개체번호"]))
                    if "개체번호" in g_df.columns else len(g_df)
                )
                st.write(f"**총 개체 수**: {n_ind}개체")
                if "time" not in g_df.columns:
                    st.write("**측정 기간**: 생육 결과 기준 (환경 데이터 참고)")
                else:
                    tmin = g_df["time"].min()
                    tmax = g_df["time"].max()
                    if pd.notna(tmin) and pd.notna(tmax):
                        st.write(
                            f"**측정 기간**: {tmin.strftime('%Y-%m-%d')} ~ {tmax.strftime('%Y-%m-%d')}"
                        )
                avail_items = [
                    c for c in [
                        "잎 수", "지상부 길이", "지하부 길이",
                        "생중량", "지상부 질량", "지하부 질량",
                    ]
                    if c in g_df.columns
                ]
                st.write(
                    f"**측정 항목**: {', '.join(avail_items) if avail_items else '없음'}"
                )
            else:
                st.info("📎 생육 결과 CSV를 업로드하면 여기에 요약이 표시됩니다.")

            if env_file is not None:
                e_df = load_env_df(env_file)
                if "time" in e_df.columns and e_df["time"].notna().any():
                    tmin_e = e_df["time"].min()
                    tmax_e = e_df["time"].max()
                    if pd.notna(tmin_e) and pd.notna(tmax_e):
                        st.write(
                            f"**환경 측정 기간**: {tmin_e.strftime('%Y-%m-%d')} ~ {tmax_e.strftime('%Y-%m-%d')}"
                        )
                cols_env = [
                    c for c in ["temperature", "humidity", "ph", "ec"]
                    if c in e_df.columns
                ]
                st.write(
                    f"**환경 측정 항목**: {', '.join(cols_env) if cols_env else '없음'}"
                )
            else:
                st.info("📎 환경 데이터 CSV를 업로드하면 여기에 요약이 표시됩니다.")

        st.divider()
        st.subheader("주요 지표")
        m1, m2, m3, m4 = st.columns(4)

        if growth_file is not None:
            g_df = load_growth_df(growth_file)
            if len(g_df) > 0:
                n_ind = (
                    len(g_df.drop_duplicates(subset=["개체번호"]))
                    if "개체번호" in g_df.columns else len(g_df)
                )
                m1.metric("총 개체수", f"{n_ind}개체")
            else:
                m1.metric("총 개체수", "—")
        else:
            m1.metric("총 개체수", "—")

        if env_file is not None:
            e_df = load_env_df(env_file)
            if "temperature" in e_df.columns:
                tc = e_df["temperature"].dropna()
                if len(tc) >= 2:
                    m2.metric("평균 온도", f"{tc.mean():.1f}℃")
                else:
                    m2.metric("평균 온도", "데이터 부족")
            else:
                m2.metric("평균 온도", "—")
        else:
            m2.metric("평균 온도", "—")

        if env_file is not None:
            e_df = load_env_df(env_file)
            if "humidity" in e_df.columns:
                hc = e_df["humidity"].dropna()
                if len(hc) >= 2:
                    m3.metric("평균 습도", f"{hc.mean():.1f}%")
                else:
                    m3.metric("평균 습도", "데이터 부족")
            else:
                m3.metric("평균 습도", "—")
        else:
            m3.metric("평균 습도", "—")

        if growth_file is not None:
            g_df = load_growth_df(growth_file)
            if "생중량" in g_df.columns:
                gwc = g_df["생중량"].dropna()
                if len(gwc) >= 2:
                    m4.metric("평균 생중량", f"{gwc.mean():.2f}g")
                else:
                    m4.metric("평균 생중량", "데이터 부족")
            else:
                m4.metric("평균 생중량", "—")
        else:
            m4.metric("평균 생중량", "—")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# 탭 2 — 환경 데이터
# ═══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    try:
        if env_file is None:
            st.info(
                "🌡️ 환경 데이터 CSV를 사이드바에서 업로드하면 이 탭의 내용이 표시됩니다."
            )
        else:
            env_df = load_env_df(env_file)

            st.subheader("환경 요약 (평균 · 최소 · 최대 · CV)")
            env_cols = [
                c for c in ["temperature", "humidity", "ph", "ec"]
                if c in env_df.columns
            ]
            if env_cols:
                summary_rows = []
                for c in env_cols:
                    s = env_df[c].dropna()
                    if len(s) >= 2:
                        cv_val = (s.std() / s.mean() * 100) if s.mean() != 0 else 0
                        summary_rows.append({
                            "측정 항목": c,
                            "평균": round(s.mean(), 2),
                            "최소": round(s.min(), 2),
                            "최대": round(s.max(), 2),
                            "표준편차": round(s.std(), 2),
                            "CV(%)": round(cv_val, 2),
                        })
                if summary_rows:
                    summary_df = pd.DataFrame(summary_rows)
                    label_map = {
                        "temperature": "온도(℃)",
                        "humidity": "습도(%)",
                        "ph": "pH",
                        "ec": "EC(mS/cm)",
                    }
                    summary_df["측정 항목"] = (
                        summary_df["측정 항목"].map(label_map).fillna(
                            summary_df["측정 항목"]
                        )
                    )
                    st.dataframe(summary_df, width="stretch")
                else:
                    st.info(
                        "환경 데이터의 숫자 값이 충분하지 않아 요약 표를 표시할 수 없습니다."
                    )
            else:
                st.info(
                    "환경 데이터에서 온도·습도·pH·EC 열을 찾을 수 없습니다. 열 이름이 다른 형식일 수 있습니다."
                )

            st.divider()
            st.subheader("날짜별 환경 변화 (일별 평균)")

            if "time" in env_df.columns:
                env_df_daily = env_df.copy()
                env_df_daily["date"] = env_df_daily["time"].dt.date
                daily = env_df_daily.groupby("date").mean(numeric_only=True).reset_index()

                chart_cols = [
                    c for c in ["temperature", "humidity", "ph", "ec"]
                    if c in daily.columns
                ]
                if chart_cols:
                    n_charts = len(chart_cols)
                    fig_env = make_subplots(
                        rows=n_charts,
                        cols=1,
                        subplot_titles=[
                            {
                                "temperature": "온도 변화",
                                "humidity": "습도 변화",
                                "ph": "pH 변화",
                                "ec": "EC 변화",
                            }[c]
                            for c in chart_cols
                        ],
                        vertical_spacing=0.12,
                    )
                    colors = {
                        "temperature": "#e74c3c",
                        "humidity": "#3498db",
                        "ph": "#27ae60",
                        "ec": "#8e44ad",
                    }
                    for i, c in enumerate(chart_cols, start=1):
                        fig_env.add_trace(
                            go.Scatter(
                                x=daily["date"],
                                y=daily[c].dropna(),
                                mode="lines",
                                name={
                                    "temperature": "온도",
                                    "humidity": "습도",
                                    "ph": "pH",
                                    "ec": "EC",
                                }[c],
                                line=dict(color=colors.get(c, "#333333"), width=2),
                            ),
                            row=i,
                            col=1,
                        )

                    fig_env.update_layout(
                        height=200 * n_charts + 50,
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        showlegend=False,
                        hovermode="x unified",
                    )
                    fig_env.update_xaxes(title_text="날짜", row=n_charts, col=1)
                    for i, c in enumerate(chart_cols, start=1):
                        fig_env.update_yaxes(
                            title_text={
                                "temperature": "온도(℃)",
                                "humidity": "습도(%)",
                                "ph": "pH",
                                "ec": "EC(mS/cm)",
                            }[c],
                            row=i,
                            col=1,
                        )
                    st.plotly_chart(fig_env, width="stretch")
                else:
                    st.info(
                        "일별 평균 그래프를 그릴 수 있는 환경 데이터(온도·습도·pH·EC)가 없습니다."
                    )
            else:
                st.info(
                    "환경 데이터에 시간(date/time) 열이 없어 시계열 그래프를 표시할 수 없습니다."
                )

            st.divider()
            st.subheader("환경 데이터 이상치 탐지 (IQR 방법)")
            if env_cols:
                outlier_rows = []
                for c in env_cols:
                    s = env_df[c].dropna()
                    if len(s) >= 4:
                        q1 = s.quantile(0.25)
                        q3 = s.quantile(0.75)
                        iqr = q3 - q1
                        lb = q1 - 1.5 * iqr
                        ub = q3 + 1.5 * iqr
                        n_out = ((s < lb) | (s > ub)).sum()
                        outlier_rows.append({
                            "측정 항목": c,
                            "Q1": round(q1, 2),
                            "Q3": round(q3, 2),
                            "IQR": round(iqr, 2),
                            "하한": round(lb, 2),
                            "상한": round(ub, 2),
                            "이상치 개수": int(n_out),
                            "전체 대비(%)": round(n_out / len(s) * 100, 2),
                        })
                if outlier_rows:
                    outlier_df = pd.DataFrame(outlier_rows)
                    outlier_df["측정 항목"] = (
                        outlier_df["측정 항목"].map(label_map).fillna(
                            outlier_df["측정 항목"]
                        )
                    )
                    st.dataframe(outlier_df, width="stretch")
                else:
                    st.info("이상치를 계산할 만큼 충분한 데이터가 없습니다.")
            else:
                st.info("이상치 탐지에 사용할 환경 측정 열이 없습니다.")

            st.divider()
            st.subheader("환경 데이터별 CV(변동계수) 비교")
            if env_cols:
                cv_vals = []
                cv_names = []
                for c in env_cols:
                    s = env_df[c].dropna()
                    if len(s) >= 2 and s.mean() != 0:
                        cv_vals.append(s.std() / s.mean() * 100)
                        cv_names.append({
                            "temperature": "온도",
                            "humidity": "습도",
                            "ph": "pH",
                            "ec": "EC",
                        }[c])
                if cv_vals:
                    fig_cv = go.Figure()
                    fig_cv.add_trace(
                        go.Bar(
                            x=cv_names,
                            y=cv_vals,
                            marker_color=(
                                ["#e74c3c", "#3498db", "#27ae60", "#8e44ad"][
                                    :len(cv_vals)
                                ]
                            ),
                            text=[f"{v:.1f}%" for v in cv_vals],
                            textposition="outside",
                            hovertemplate=(
                                "측정 항목: %{{x}}<br>CV: %{{y:.2f}}%<extra></extra>"
                            ),
                        )
                    )
                    fig_cv.update_layout(
                        height=400,
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        xaxis_title="측정 항목",
                        yaxis_title="CV (%)",
                        showlegend=False,
                    )
                    st.plotly_chart(fig_cv, width="stretch")
                    st.caption(
                        "CV가 높을수록 측정값이 평균 주변에서 크게 변동함을 의미합니다."
                    )
                else:
                    st.info("CV 계산이 가능한 환경 데이터가 충분하지 않습니다.")
            else:
                st.info("환경 데이터에 CV를 계산할 측정 열이 없습니다.")

            with st.expander("📄 환경 데이터 원본 테이블 및 CSV 다운로드"):
                st.dataframe(env_df, width="stretch")
                st.download_button(
                    label="📥 환경 데이터 CSV 다운로드 (utf-8-sig)",
                    data=make_download_csv(env_df),
                    file_name=f"{school_name}_환경데이터.csv",
                    mime="text/csv",
                )

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# 탭 3 — 생육 결과
# ═══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    try:
        if growth_file is None:
            st.info(
                "📊 생육 결과 CSV를 사이드바에서 업로드하면 이 탭의 내용이 표시됩니다."
            )
        else:
            growth_df = load_growth_df(growth_file)

            st.header("📊 생육 결과 분석")

            if "생중량" in growth_df.columns:
                gwc = growth_df["생중량"].dropna()
                if len(gwc) >= 2:
                    avg_gw = gwc.mean()
                    std_gw = gwc.std()
                    cv_gw = (std_gw / avg_gw * 100) if avg_gw > 0 else 0
                    min_idx = gwc.idxmin()
                    max_idx = gwc.idxmax()
                    min_val = gwc.min()
                    max_val = gwc.max()

                    st.subheader("🥇 핵심 결과")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("평균 생중량", f"{avg_gw:.2f}g")
                    c2.metric(
                        "최대 생중량 개체",
                        f"개체 {growth_df.loc[max_idx, '개체번호']} — {max_val:.2f}g",
                    )
                    c3.metric(
                        "최소 생중량 개체",
                        f"개체 {growth_df.loc[min_idx, '개체번호']} — {min_val:.2f}g",
                    )

                    st.info(
                        f"""
**생중량 분포 요약**
- 평균: {avg_gw:.2f}g
- 표준편차: {std_gw:.2f}g
- 변동계수(CV): {cv_gw:.1f}%
- 최소: {min_val:.2f}g (개체 {growth_df.loc[min_idx, '개체번호']})
- 최대: {max_val:.2f}g (개체 {growth_df.loc[max_idx, '개체번호']})
"""
                    )

            st.divider()
            st.subheader("개체별 생육 비교 (막대 그래프)")

            bar_metrics = [
                c for c in ["생중량", "잎 수", "지상부 길이", "지하부 길이"]
                if c in growth_df.columns
            ]
            if bar_metrics:
                n_bar = len(bar_metrics)
                fig_bar = make_subplots(
                    rows=n_bar,
                    cols=1,
                    subplot_titles=[f"평균 {m}별 개체" for m in bar_metrics],
                    vertical_spacing=0.10,
                )
                sorted_df = (
                    growth_df.sort_values("개체번호", ascending=True)
                    if "개체번호" in growth_df.columns else growth_df
                )

                bar_colors_map = {
                    "생중량": "#27ae60",
                    "잎 수": "#2980b9",
                    "지상부 길이": "#e67e22",
                    "지하부 길이": "#8e44ad",
                }

                for i, metric in enumerate(bar_metrics, start=1):
                    vals = sorted_df[metric].dropna()
                    if len(vals) < 2:
                        st.info(f"{metric} 값이 충분하지 않아 그래프를 표시할 수 없습니다.")
                        continue
                    x_vals = (
                        sorted_df.loc[vals.index, "개체번호"]
                        if "개체번호" in sorted_df.columns else vals.index
                    )
                    threshold_high = vals.quantile(0.75)
                    threshold_low = vals.quantile(0.25)

                    colors_bar = []
                    for v in vals:
                        if v >= threshold_high:
                            colors_bar.append("#27ae60")
                        elif v <= threshold_low:
                            colors_bar.append("#e74c3c")
                        else:
                            colors_bar.append(bar_colors_map.get(metric, "#3498db"))

                    fig_bar.add_trace(
                        go.Bar(
                            x=x_vals,
                            y=vals,
                            name=metric,
                            marker_color=colors_bar,
                            hovertemplate=(
                                f"{metric}<br>개체: %{{x}}<br>값: %{{y:.2f}}<extra></extra>"
                            ),
                        ),
                        row=i,
                        col=1,
                    )

                fig_bar.update_layout(
                    height=180 * n_bar + 60,
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    showlegend=False,
                    hovermode="x unified",
                )
                for i, metric in enumerate(bar_metrics, start=1):
                    unit = ""
                    if metric == "생중량":
                        unit = "(g)"
                    elif metric == "잎 수":
                        unit = "(장)"
                    elif "길이" in metric:
                        unit = "(mm)"
                    fig_bar.update_yaxes(title_text=f"{metric}{unit}", row=i, col=1)

                st.plotly_chart(fig_bar, width="stretch")
                st.caption(
                    "🟢 초록 막대: 상위 25%  |  🔴 빨강 막대: 하위 25%  |  🔵 파란 막대: 중간 구간"
                )
            else:
                st.info(
                    "생육 데이터에서 막대 그래프로 표시할 측정 항목(생중량·잎 수·지상부 길이·지하부 길이)을 찾을 수 없습니다."
                )

            st.divider()
            st.subheader("생중량 분포 (박스플롯)")
            if "생중량" in growth_df.columns:
                gwb = growth_df["생중량"].dropna()
                if len(gwb) >= 2:
                    fig_box = go.Figure()
                    fig_box.add_trace(
                        go.Box(
                            y=gwb,
                            name="생중량",
                            boxpoints="all",
                            jitter=0.3,
                            pointpos=-1.8,
                            marker=dict(color="#27ae60", size=4),
                            line=dict(color="#1e8449"),
                        )
                    )
                    fig_box.update_layout(
                        height=400,
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        yaxis_title="생중량(g)",
                        showlegend=False,
                    )
                    st.plotly_chart(fig_box, width="stretch")
                else:
                    st.info("생중량 값이 충분하지 않아 박스플롯을 표시할 수 없습니다.")
            else:
                st.info("생육 데이터에 생중량 열이 없어 박스플롯을 표시할 수 없습니다.")

            with st.expander("📄 생육 데이터 원본 테이블 및 CSV 다운로드"):
                st.dataframe(growth_df, width="stretch")
                st.download_button(
                    label="📥 생육 결과 CSV 다운로드 (utf-8-sig)",
                    data=make_download_csv(growth_df),
                    file_name=f"{school_name}_생육결과데이터.csv",
                    mime="text/csv",
                )

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# 탭 4 — 가설 검증
# ═══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    try:
        if growth_file is None:
            st.info("🔬 생육 결과 CSV를 업로드하면 가설 검증 탭의 내용이 표시됩니다.")
        else:
            growth_df = load_growth_df(growth_file)

            st.header("🔬 가설 검증 — 지상부 vs 지하부 길이")

            above_col = "지상부 길이" if "지상부 길이" in growth_df.columns else None
            below_col = "지하부 길이" if "지하부 길이" in growth_df.columns else None

            if above_col is None or below_col is None:
                st.info(
                    "가설 검증을 하려면 생육 데이터에 '지상부 길이'와 '지하부 길이' 열이 모두 있어야 합니다."
                )
            else:
                df_ab = growth_df[[above_col, below_col]].dropna()
                if len(df_ab) < 2:
                    st.info("지상부·지하부 길이 데이터가 충분하지 않아 가설 검증을 건너뜁니다.")
                else:
                    x_vals = df_ab[above_col].values
                    y_vals = df_ab[below_col].values
                    r = np.corrcoef(x_vals, y_vals)[0, 1]
                    slope, intercept = np.polyfit(x_vals, y_vals, 1)

                    st.subheader("🥇 핵심 결과 카드")
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("지상부 평균 길이", f"{x_vals.mean():.1f}mm")
                    k2.metric("지하부 평균 길이", f"{y_vals.mean():.1f}mm")
                    k3.metric(
                        "지상부:지하부 평균 비율",
                        f"{x_vals.mean() / y_vals.mean():.3f}",
                    )
                    k4.metric("상관계수 r", f"{r:.4f}")

                    st.info(
                        f"""
**지상부 vs 지하부 길이 상관관계 요약**
- 지상부 평균 길이: {x_vals.mean():.1f}mm (n={len(x_vals)})
- 지하부 평균 길이: {y_vals.mean():.1f}mm (n={len(y_vals)})
- 지상부:지하부 평균 비율: {x_vals.mean() / y_vals.mean():.3f}
- 상관계수 r = {r:.4f}
  → {'강한 양의 상관' if r > 0.7 else '중간 양의 상관' if r > 0.4 else '약한 상관' if abs(r) < 0.4 else '중간 음의 상관' if r < -0.4 else '강한 음의 상관'}
- 추세선: y = {slope:.3f}x + {intercept:.3f}

> **해석 안내**
> 상관계수가 0에 가까우면 지상부 길이와 지하부 길이 사이에 뚜렷한 선형 관계가 없음을 의미합니다.
> 즉, 지상부가 긴 개체가 반드시 지하부도 긴 것은 아니며, 개체별로 지상부·지하부 생장 패턴이 다를 수 있습니다.
"""
                    )

                    # ── 개체별 지상부-지하부 그룹 막대 그래프 ──────────────
                    st.divider()
                    st.subheader("개체별 지상부·지하부 길이 (grouped bar — 한 그래프에 나란히)")

                    sorted_ab = (
                        growth_df.sort_values("개체번호", ascending=True)
                        if "개체번호" in growth_df.columns else growth_df
                    )
                    x_labels = (
                        sorted_ab["개체번호"].values
                        if "개체번호" in sorted_ab.columns
                        else list(range(len(sorted_ab)))
                    )

                    fig_gbar = go.Figure()
                    fig_gbar.add_trace(
                        go.Bar(
                            x=x_labels,
                            y=sorted_ab[above_col],
                            name="지상부 길이",
                            marker_color="#e67e22",
                            hovertemplate=(
                                "지상부 길이<br>개체: %{{x}}<br>길이: %{{y:.1f}}mm<extra></extra>"
                            ),
                        )
                    )
                    fig_gbar.add_trace(
                        go.Bar(
                            x=x_labels,
                            y=sorted_ab[below_col],
                            name="지하부 길이",
                            marker_color="#3498db",
                            hovertemplate=(
                                "지하부 길이<br>개체: %{{x}}<br>길이: %{{y:.1f}}mm<extra></extra>"
                            ),
                        )
                    )

                    fig_gbar.update_layout(
                        height=460,
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        xaxis_title="개체번호",
                        yaxis_title="길이 (mm)",
                        barmode="group",
                        bargap=0.15,
                        bargroupgap=0.05,
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                        ),
                        hovermode="x unified",
                    )

                    st.plotly_chart(fig_gbar, width="stretch")
                    st.caption(
                        "🟠 주황색: 지상부 길이  |  🔵 파란색: 지하부 길이 — 개체마다 두 값이 나란히 표시됩니다."
                    )

                    # ── 지상부:지하부 평균 비율 원형 그래프 ─────────────────
                    st.divider()
                    st.subheader("지상부 : 지하부 평균 길이 비율 (원형 그래프)")

                    av_mean = x_vals.mean()
                    bel_mean = y_vals.mean()
                    ratio_val = av_mean / bel_mean if bel_mean != 0 else 0

                    fig_pie = go.Figure()
                    fig_pie.add_trace(
                        go.Pie(
                            labels=["지상부 길이", "지하부 길이"],
                            values=[av_mean, bel_mean],
                            marker_colors=["#e67e22", "#3498db"],
                            textinfo="label+percent+value",
                            hovertemplate=(
                                "부분: %{{label}}<br>평균 길이: %{{value:.1f}}mm<br>비율: %{{percent}<extra></extra>"
                            ),
                        )
                    )
                    fig_pie.update_layout(
                        height=420,
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        title=f"지상부 : 지하부 평균 길이 비율 = {ratio_val:.3f} : 1",
                        showlegend=True,
                    )
                    st.plotly_chart(fig_pie, width="stretch")

                    # ── 지상부-지하부 길이 상관관계 산점도 + 추세선 ─────────
                    st.divider()
                    st.subheader("지상부 길이 vs 지하부 길이 — 상관관계 산점도")

                    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                    y_line = slope * x_line + intercept

                    fig_sc = go.Figure()
                    fig_sc.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            mode="markers",
                            name="개체별 데이터",
                            marker=dict(
                                size=8,
                                color="#2980b9",
                                opacity=0.7,
                                line=dict(width=1, color="#2c3e50"),
                            ),
                            hovertemplate=(
                                "지상부: %{{x:.1f}}mm<br>지하부: %{{y:.1f}}mm<extra></extra>"
                            ),
                        )
                    )
                    fig_sc.add_trace(
                        go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode="lines",
                            name=f"추세선 (r={r:.4f})",
                            line=dict(color="#e74c3c", width=2, dash="dash"),
                        )
                    )

                    fig_sc.update_layout(
                        height=450,
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        xaxis_title="지상부 길이 (mm)",
                        yaxis_title="지하부 길이 (mm)",
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                        ),
                    )

                    st.plotly_chart(fig_sc, width="stretch")

                    # ── 지상부-지하부 길이 상관관계 히트맵 ─────────────────
                    st.divider()
                    st.subheader("지상부·지하부 길이 상관관계 히트맵")

                    heat_df = df_ab[[above_col, below_col]].copy()
                    heat_df.columns = ["지상부 길이", "지하부 길이"]
                    corr_matrix = heat_df.corr()

                    z_values = corr_matrix.values
                    x_labels_h = list(corr_matrix.columns)
                    y_labels_h = list(corr_matrix.index)

                    colorscale = [
                        [0.0, "#0d47a1"],
                        [0.5, "#ffffff"],
                        [1.0, "#b71c1c"],
                    ]

                    fig_heat = go.Figure()
                    fig_heat.add_trace(
                        go.Heatmap(
                            z=z_values,
                            x=x_labels_h,
                            y=y_labels_h,
                            colorscale=colorscale,
                            zmin=-1.0,
                            zmax=1.0,
                            text=np.round(z_values, 4),
                            texttemplate="%{text:.4f}",
                            textfont=dict(
                                size=18,
                                color="#222222",
                                family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                            ),
                            hoverongaps=False,
                            hovertemplate=(
                                "%{y} vs %{x}<br>상관계수 r = %{z:.4f}<extra></extra>"
                            ),
                            colorbar=dict(
                                title="상관계수 r",
                            ),
                        )
                    )

                    fig_heat.update_layout(
                        height=430,
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        xaxis_title="비교 변수",
                        yaxis_title="비교 변수",
                        xaxis=dict(side="top"),
                        yaxis=dict(
                            autorange="reversed",
                            tickfont=dict(
                                family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                            ),
                        ),
                        xaxis_tickfont=dict(
                            family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                        ),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=60, r=80, t=30, b=60),
                    )

                    st.plotly_chart(fig_heat, width="stretch")

                    st.info(
                        f"""
**히트맵 해석**

- **대각선(1.0000)**: 같은 변수끼리의 상관관계는 항상 1 (자기 자신과 완전히 같음)
- **비대각선 r = {r:.4f}**: 지상부 길이와 지하부 길이 사이의 상관계수
  - 0에 가까움 → 두 길이 사이에 뚜렷한 선형 관계가 **없음**
  - 색깔도 흰색에 가까움 → 상관이 약함을 시각적으로 확인 가능
- 색이 **파랑**에 가까우면 음의 상관, **빨강**에 가까우면 양의 상관을 의미합니다.
"""
                    )

                    # ── 가설 검증 결과 분석 설명 ─────────────────────────────
                    st.divider()
                    st.subheader("📝 가설 검증 결과 분석")

                    if abs(r) < 0.3:
                        corr_desc = "매우 약한 상관관계"
                    elif abs(r) < 0.5:
                        corr_desc = "약한 상관관계"
                    elif abs(r) < 0.7:
                        corr_desc = "중간 정도의 상관관계"
                    else:
                        corr_desc = "강한 상관관계"

                    direction = "양의" if r > 0 else "음의"

                    st.markdown(
                        f"""
**검증 가설**: "지상부 길이가 긴 개체는 지하부 길이도 길 것이다." (지상부 길이와 지하부 길이 사이에 양의 상관관계가 있을 것이다.)

**검증 결과**
- 상관계수 r = {r:.4f} → {direction} {corr_desc}
- 지상부 평균 길이: {x_vals.mean():.1f}mm
- 지하부 평균 길이: {y_vals.mean():.1f}mm
- 지상부:지하부 평균 비율: {ratio_val:.3f}

**분석 설명**

산점도, 추세선, 히트맵을 종합하여 볼 때, 지상부 길이와 지하부 길이 사이의 선형 관계는 **{corr_desc}** 수준입니다.

- 상관계수가 0에 가까우므로, 지상부 길이가 길다고 해서 지하부 길이도 반드시 길다고 단정하기 어렵습니다.
- 일부 개체는 지상부는 짧지만 지하부가 매우 길고, 반대의 패턴을 보이는 개체도 있어 **개체별로 지상부·지하부 생장 전략이 다를 가능성**이 있습니다.
- 히트맵에서 비대각선 색이 흰색에 가까운 것은 두 변수 간 상관이 약함을 시각적으로 보여줍니다.
- 원형 그래프에서 보듯, 평균적으로 지상부와 지하부 길이는 {ratio_val:.3f} : 1 비율로 나타나며, 이는 지상부가 지하부보다 {((ratio_val - 1) * 100):.1f}% {'김' if ratio_val > 1 else '짧음'}을 의미합니다.

> **참고**: 상관계수는 선형 관계만 나타내므로, 실제 관계는 곡선 형태일 수도 있습니다. 또한 표본이 81개체로 제한되어 있어 이 결과를 일반화하려면 더 많은 데이터가 필요합니다.
"""
                    )

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# 탭 5 — 한계 및 개선방향
# ═══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    try:
        st.header("💡 연구의 한계 및 개선 방향")

        st.subheader("🔍 연구 한계")

        st.markdown(
            """
**1. 생육 측정일 정보 부재**
생육 결과 데이터에는 생육 측정일(날짜·시각)이 기록되어 있지 않아, 환경 데이터의 측정 시점과 생육 결과를 같은 시점에 연결하여 분석하기 어렵습니다. 환경 조건은 시간에 따라 변하므로, 어떤 환경에서 측정되었는지 알 수 없으면 인과적 해석이 제한됩니다.

**2. 개체 위치·처리구 정보 부재**
개체의 위치, 화분 번호, 재배대 위치, 광 조건, 처리구 등의 정보가 없어 개체 간 환경 차이를 반영한 분석이 불가능합니다. 같은 환경에서도 위치별로 미세기후가 다를 수 있으므로, 이러한 정보가 있으면 더 정밀한 비교가 가능합니다.

**3. 제한된 표본 수**
생육 결과 데이터의 표본은 81개체로, 상관관계를 일반화하거나 통계적으로 안정적인 결론을 내리기에는 제한적입니다. 표본 수가 작으면 우연에 의한 상관관계가 나타날 가능성도 있습니다.

**4. 환경 데이터 측정 시각 불일치**
환경 데이터의 측정 시각이 일정하지 않아, 생육 측정 시점과 환경 조건을 정확히 대응시키기 어렵습니다. 일별 평균을 사용하더라도, 실제 생육 측정 시점의 조건과 다를 수 있습니다.
"""
        )

        st.divider()
        st.subheader("🚀 개선 방향")

        st.markdown(
            """
**1. 생육 측정일 기록**
동일한 실험을 다시 한다면 생육 측정일(날짜와 시각)을 개체번호와 함께 반드시 기록합니다. 이를 통해 환경 데이터와 생육 결과를 시간 축으로 정확히 연결할 수 있습니다.

**2. 개체 위치·환경 정보 기록**
개체의 위치를 화분 번호, 재배대 위치, 광 조건 등과 함께 기록하여 개체 간 환경 차이를 설명할 수 있게 합니다. 예를 들어 광량이 많은 위치의 개체와 적은 위치의 개체를 구분하여 비교할 수 있습니다.

**3. 측정 시점 일치**
환경 데이터 측정과 생육 측정 시점을 일치시키는 방식으로 두 데이터를 연계하여 분석합니다. 생육 측정 직전·후의 환경 조건을 직접 활용할 수 있습니다.

**4. 표본 수 확대 및 반복 실험**
표본 수를 늘려 여러 개체를 반복 측정하고, 가능하면 실험을 2~3회 반복하여 결과의 신뢰성을 높입니다. 계절이나 조건별로 반복하면 더 일반화된 결론을 얻을 수 있습니다.

**5. 환경 요인과 생육 결과의 통합 분석**
지상부 길이와 지하부 길이뿐 아니라 온도·습도·pH·EC 등 환경 요인과 생육 결과를 함께 비교·분석하여, 줄기와 뿌리 생장에 영향을 주는 환경 요인을 탐색합니다. 예를 들어 온도가 높은 날 측정한 개체의 생육이 더 좋은지 등을 검토할 수 있습니다.
"""
        )

        st.divider()
        st.caption(
            "본 대시보드는 업로드된 CSV 데이터를 기반으로 한 탐색적 분석 도구입니다. 통계적 해석은 표본과 데이터의 한계를 고려하여 참고용으로만 활용하세요."
        )

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
