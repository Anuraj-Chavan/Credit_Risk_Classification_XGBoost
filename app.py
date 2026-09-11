import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title='Credit Risk Prediction', page_icon='💳', layout='wide')
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'xgboost_credit_risk_model.pkl'
ENCODER_PATH = BASE_DIR / 'xgboost_label_encoder.pkl'
FEATURES_PATH = BASE_DIR / 'xgboost_feature_names.pkl'

@st.cache_resource
def load_artifacts():
    return (joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH), joblib.load(FEATURES_PATH))

missing = [p.name for p in (MODEL_PATH, ENCODER_PATH, FEATURES_PATH) if not p.exists()]
if missing:
    st.error('Required model files are missing.')
    st.write('Keep these files in the same folder as app.py:')
    for f in missing: st.write(f'- `{f}`')
    st.stop()

model, label_encoder, feature_names = load_artifacts()

st.title('💳 Credit Risk Prediction App')
st.write('Enter applicant information to predict the credit risk category using the trained XGBoost model.')
st.info('The input is converted to the same 65-feature structure used by the trained model.')

categorical_features = ['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
categorical_dummy_features = [f for f in feature_names if any(f.startswith(c + '_') for c in categorical_features)]
numeric_features = [f for f in feature_names if f not in categorical_dummy_features and f != 'EDUCATION']

def categories(prefix):
    return [f[len(prefix)+1:] for f in feature_names if f.startswith(prefix + '_')]

with st.form('credit_risk_form'):
    st.subheader('👤 Applicant Information')
    c1, c2, c3 = st.columns(3)
    with c1: marital_status = st.selectbox('Marital Status', categories('MARITALSTATUS'))
    with c2: gender = st.selectbox('Gender', categories('GENDER'))
    with c3: education = st.selectbox('Education', ['SSC','12TH','GRADUATE','UNDER GRADUATE','POST-GRADUATE','OTHERS','PROFESSIONAL'])

    st.subheader('🏦 Product & Enquiry Information')
    c1, c2 = st.columns(2)
    with c1: last_prod = st.selectbox('Last Product Enquiry', categories('last_prod_enq2'))
    with c2: first_prod = st.selectbox('First Product Enquiry', categories('first_prod_enq2'))

    st.subheader('📊 Financial & Credit Information')
    values = {}
    for start in range(0, len(numeric_features), 3):
        cols = st.columns(3)
        for col, feature in zip(cols, numeric_features[start:start+3]):
            if feature == 'AGE': default, minimum = 30.0, 18.0
            elif feature == 'Credit_Score': default, minimum = 700.0, 300.0
            elif feature == 'NETMONTHLYINCOME': default, minimum = 30000.0, 0.0
            else: default, minimum = 0.0, 0.0
            with col:
                values[feature] = st.number_input(feature.replace('_',' '), min_value=minimum, value=default, step=1.0, format='%.2f')

    submitted = st.form_submit_button('🔍 Predict Credit Risk', use_container_width=True)

if submitted:
    education_mapping = {'SSC':1, '12TH':2, 'GRADUATE':3, 'UNDER GRADUATE':3, 'POST-GRADUATE':4, 'OTHERS':1, 'PROFESSIONAL':3}
    input_df = pd.DataFrame(0.0, index=[0], columns=feature_names)
    for feature in numeric_features: input_df.loc[0, feature] = values[feature]
    input_df.loc[0, 'EDUCATION'] = education_mapping[education]
    selected = {'MARITALSTATUS':marital_status, 'GENDER':gender, 'last_prod_enq2':last_prod, 'first_prod_enq2':first_prod}
    for feature, value in selected.items():
        dummy = f'{feature}_{value}'
        if dummy not in input_df.columns:
            st.error(f'Category {value} is not present in the trained feature structure for {feature}.')
            st.stop()
        input_df.loc[0, dummy] = 1
    input_df = input_df[feature_names]

    try:
        encoded_pred = int(model.predict(input_df)[0])
        prediction = label_encoder.inverse_transform([encoded_pred])[0]
        probabilities = model.predict_proba(input_df)[0]
        st.divider()
        st.subheader('🎯 Prediction Result')
        if prediction in ('P1','P2'): st.success(f'Predicted Credit Risk Category: **{prediction}**')
        elif prediction == 'P3': st.warning(f'Predicted Credit Risk Category: **{prediction}**')
        elif prediction == 'P4': st.error(f'Predicted Credit Risk Category: **{prediction}**')
        else: st.info(f'Predicted Credit Risk Category: **{prediction}**')
        result = pd.DataFrame({'Credit Risk Category': label_encoder.classes_, 'Probability': (probabilities*100).round(2)})
        result['Probability'] = result['Probability'].astype(str) + '%'
        st.subheader('Prediction Probabilities')
        st.dataframe(result, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error('An error occurred while making the prediction.')
        st.exception(e)

with st.sidebar:
    st.header('About the Model')
    st.write('XGBoost multiclass classifier trained for the Credit Risk Classification project.')
    st.write(f'**Model features:** {len(feature_names)}')
    st.write('**Target classes:** ' + ', '.join(label_encoder.classes_))
