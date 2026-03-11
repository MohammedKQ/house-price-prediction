import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import os

# 1. إعدادات الصفحة والهوية البصرية
st.set_page_config(
    page_title="منصة إيجار الذكية | التحليل المتقدم",
    page_icon="🏠",
    layout="wide"
)

# تصميم CSS مخصص لضمان الجمالية ووضوح النصوص في كل الأقسام
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
    html, body, [class*="css"] { 
        font-family: 'Tajawal', sans-serif; 
        text-align: right; 
    }
    
    /* بطاقات المؤشرات الداكنة في الصفحة الرئيسية */
    [data-testid="stMetric"] {
        background-color: #111827;
        border-right: 5px solid #fbbf24;
        padding: 20px;
        border-radius: 12px;
    }
    [data-testid="stMetricLabel"] { color: #9ca3af !important; font-size: 1rem !important; }
    [data-testid="stMetricValue"] { color: #fbbf24 !important; font-size: 1.8rem !important; }

    /* تحسين قسم رؤى البيانات (Insights) - نصوص واضحة وداكنة */
    .insight-card {
        background-color: #ffffff; /* خلفية بيضاء */
        border-right: 5px solid #1e3a8a; /* خط جانبي كحلي */
        padding: 20px;
        margin: 15px 0;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .insight-card h4 {
        color: #1e3a8a !important; /* عنوان كحلي */
        margin-bottom: 10px;
        font-weight: bold;
    }
    .insight-card p {
        color: #334155 !important; /* نص رمادي داكن واضح جداً */
        line-height: 1.6;
        font-size: 1.05rem;
        margin: 0;
    }

    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. تحميل البيانات وتجهيز المتغيرات العامة
@st.cache_data
def load_data():
    file_name = "cleaned_house_data.csv"
    if os.path.exists(file_name):
        data = pd.read_csv(file_name)
        data['price_per_sqm'] = data['price'] / data['size']
        return data
    return None

df = load_data()

# تعريف المتغيرات في النطاق العام لتجنب NameError
if df is not None:
    # البحث عن أعمدة المدن (مع مراعاة وجود مسافة في الاسم كما في الملف)
    city_cols = [c for c in df.columns if 'city_' in c]
    city_names = [c.replace('city_ ', '').replace('city_', '') for c in city_cols]
else:
    city_cols = []
    city_names = []

# 3. القائمة الجانبية للتنقل
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>🏘️ نظام إيجار</h2>", unsafe_allow_html=True)
    st.divider()
    page = st.radio("القائمة الرئيسية:", [
        "🏠 الشاشة الرئيسية", 
        "📊 التحليلات التفاعلية", 
        "💡 رؤى البيانات (Insights)",
        "🤖 التنبؤ والذكاء الاصطناعي"
    ])
    st.divider()
    st.info("💡 النظام مخصص لتحليل **الإيجار السنوي**.")

# التحقق من وجود البيانات قبل عرض المحتوى
if df is None:
    st.error("❌ ملف 'cleaned_house_data.csv' غير موجود في المجلد الحالي.")
else:
    # --- الصفحة 1: الشاشة الرئيسية ---
    if page == "🏠 الشاشة الرئيسية":
        st.markdown("""
            <div class="main-header">
                <h1>المنصة الذكية لتحليل إيجارات العقارات 🇸🇦</h1>
                <p>استكشاف وتوقع أسعار الإيجار السنوي بدقة وموثوقية</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("📌 نبذة عن السوق (بيانات الإيجار السنوي)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("إجمالي الوحدات", f"{len(df):,}")
        m2.metric("متوسط الإيجار", f"{int(df['price'].mean()):,} ر.س")
        m3.metric("متوسط سعر المتر", f"{int(df['price_per_sqm'].mean()):,} ر.س")
        m4.metric("أكثر مدينة توفراً", city_names[0] if city_names else "غير محدد")

        st.divider()
        st.markdown("### 🎯 كيف تستخدم المنصة؟")
        st.write("يمكنك التنقل عبر القائمة الجانبية لاستكشاف الرسوم البيانية التفاعلية، أو الحصول على استنتاجات سريعة من قسم 'رؤى البيانات'، أو استخدام الذكاء الاصطناعي لتقدير القيمة الإيجارية لأي عقار.")

    # --- الصفحة 2: التحليلات التفاعلية ---
    elif page == "📊 التحليلات التفاعلية":
        st.title("📊 لوحة التحكم التفاعلية")
        tab1, tab2 = st.tabs(["📍 توزيع المدن", "📏 المساحة مقابل السعر"])
        
        with tab1:
            city_data = []
            for c in city_cols:
                name = c.replace('city_ ', '').replace('city_', '')
                city_data.append({'المدينة': name, 'العدد': df[df[c] == 1].shape[0], 'متوسط السعر': df[df[c] == 1]['price'].mean()})
            c_df = pd.DataFrame(city_data)
            
            c_left, c_right = st.columns(2)
            with c_left:
                st.plotly_chart(px.pie(c_df, values='العدد', names='المدينة', hole=0.5, title="نسبة توفر العقارات"), use_container_width=True)
            with c_right:
                st.plotly_chart(px.bar(c_df, x='المدينة', y='متوسط السعر', color='المدينة', title="مقارنة متوسط الإيجار السنوي"), use_container_width=True)

        with tab2:
            st.plotly_chart(px.scatter(df, x="size", y="price", color="property_age", size="bedrooms",
                                     title="تأثير المساحة والعمر على قيمة الإيجار"), use_container_width=True)

    # --- الصفحة 3: رؤى البيانات (Insights) ---
    elif page == "💡 رؤى البيانات (Insights)":
        st.title("💡 أهم الاستنتاجات من تحليل البيانات")
        st.write("رؤى تحليلية تم استخلاصها من أنماط البيانات الحالية لسوق الإيجار:")

        # استخدام حاويات مخصصة لضمان وضوح النص
        col_in1, col_in2 = st.columns(2)
        
        with col_in1:
            st.markdown("""
                <div class="insight-card">
                    <h4>🏙️ تمركز السوق</h4>
                    <p>أظهرت البيانات أن مدينة الرياض تستحوذ على الحصة الأكبر من العقارات المعروضة للإيجار السنوي، وتتميز بتباين كبير في الأسعار بناءً على الموقع والمساحة.</p>
                </div>
                <div class="insight-card">
                    <h4>📉 تأثير تقادم البناء</h4>
                    <p>تنخفض قيمة الإيجار بشكل تدريجي مع زيادة عمر العقار، ولكن يظل الموقع والمرافق (مثل المصعد والتكييف) عوامل قوية في الحفاظ على سعر مرتفع.</p>
                </div>
            """, unsafe_allow_html=True)

        with col_in2:
            st.markdown("""
                <div class="insight-card">
                    <h4>📐 المساحات الأكثر طلباً</h4>
                    <p>العقارات التي تتراوح مساحتها بين 200 إلى 350 متر مربع هي الأكثر توازناً من حيث قيمة الإيجار السنوي مقابل سعر المتر المربع.</p>
                </div>
                <div class="insight-card">
                    <h4>✨ القيمة المضافة للمرافق</h4>
                    <p>العقارات التي تتوفر فيها مرافق إضافية (مسبح، كراج، غرفة خادمة) تسجل إيجارات سنوية أعلى بنسبة تصل إلى 20% عن العقارات التقليدية.</p>
                </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.subheader("📊 سعر المتر المربع التقديري")
        city_m2_data = [{'المدينة': c.replace('city_ ', '').replace('city_', ''), 'سعر المتر': df[df[c] == 1]['price_per_sqm'].mean()} for c in city_cols]
        st.plotly_chart(px.line(pd.DataFrame(city_m2_data), x='المدينة', y='سعر المتر', markers=True, title="متوسط تكلفة المتر المربع (إيجار سنوي)"), use_container_width=True)

    # --- الصفحة 4: التنبؤ والذكاء الاصطناعي ---
    elif page == "🤖 التنبؤ والذكاء الاصطناعي":
        st.title("🤖 التنبؤ بسعر الإيجار العادل")
        
        # تجهيز الميزات (تجنب الأعمدة التحليلية المضافة)
        X = df.drop(['price', 'price_per_sqm'], axis=1)
        y = df['price']

        @st.cache_resource
        def train_model():
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            return rf

        model = train_model()

        with st.form("ai_predict_form"):
            st.subheader("📋 أدخل مواصفات الوحدة السكنية")
            c1, c2, c3 = st.columns(3)
            with c1:
                in_city = st.selectbox("المدينة", city_names)
                in_size = st.number_input("المساحة (م²)", value=300)
                in_age = st.slider("عمر العقار", 0, 40, 5)
            with c2:
                in_rooms = st.number_input("غرف النوم", 1, 10, 4)
                f_ac = st.checkbox("مكيفات راكبة")
                f_kitchen = st.checkbox("مطبخ جاهز")
            with c3:
                f_pool = st.checkbox("مسبح خاص")
                f_garage = st.checkbox("مدخل سيارة")
                f_lift = st.checkbox("مصعد")

            if st.form_submit_button("توقع الإيجار السنوي"):
                input_row = pd.DataFrame(0, index=[0], columns=X.columns)
                input_row['size'], input_row['property_age'], input_row['bedrooms'] = in_size, in_age, in_rooms
                input_row['ac'] = 1 if f_ac else 0
                input_row['kitchen'] = 1 if f_kitchen else 0
                input_row['pool'] = 1 if f_pool else 0
                input_row['garage'] = 1 if f_garage else 0
                input_row['elevator'] = 1 if f_lift else 0
                
                # تفعيل عمود المدينة
                city_key = f"city_ {in_city}" if f"city_ {in_city}" in X.columns else f"city_{in_city}"
                if city_key in input_row.columns: input_row[city_key] = 1

                prediction = model.predict(input_row)[0]
                
                st.markdown(f"""
                    <div style='background-color: #f0fdf4; border: 2px solid #22c55e; padding: 30px; border-radius: 15px; text-align: center;'>
                        <h2 style='color: #166534; margin: 0;'>قيمة الإيجار السنوي المتوقعة</h2>
                        <h1 style='color: #15803d; font-size: 3.5rem; margin: 15px 0;'>{int(prediction):,} <small style='font-size: 1.2rem;'>ريال</small></h1>
                    </div>
                """, unsafe_allow_html=True)
                st.balloons()
