import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time

import requests
from io import BytesIO


# model = joblib.load(BytesIO(response.content))
# --- Configuration and Setup ---
# pphuangkaeo/loan_rf_model/blob/main/fine_tuned_rf_model.pkl
st.set_page_config(page_title="Loan Pal Chatbot", layout="centered", initial_sidebar_state="collapsed")

# Load trained machine learning model
try:
    url = "https://huggingface.co/pphuangkaeo/loan_rf_model/resolve/main/fine_tuned_rf_model.pkl"
    response = requests.get(url)
    model = joblib.load(BytesIO(response.content))
    model_features = list(model.feature_names_in_)

except FileNotFoundError:
    st.error("Oh snap! 😥 I can't find my prediction brain ('fine_tuned_rf_model.pkl'). Please make sure it's here.")
    st.stop()
except Exception as e:
    st.error(f"Hmm, something went wrong while loading my brain: {e}")
    st.stop()

# --- Define Conceptual Features, Questions, Options, and Defaults ---

conceptual_features_info = {
    # --- Numeric Features ---
    "loan_amnt": {"type": "numeric", "question": "💰 First off, how much are you looking to borrow?", "default_example": 10000},
    "int_rate": {"type": "numeric", "question": "📈 What's the interest rate (%) offered?", "default_example": 12.5},
    "installment": {"type": "numeric", "question": "💳 Got it. What's the estimated monthly payment (installment)?", "default_example": 330.50},
    "annual_inc": {"type": "numeric", "question": "💼 Roughly, what's your total annual income?", "default_example": 65000},
    "dti": {"type": "numeric", "question": "📊 What's your debt-to-income ratio (excluding this loan)? (%)", "default_example": 18.0},
    "delinq_2yrs": {"type": "numeric", "question": "❗ Any delinquencies (30+ days late) in the last 2 years?", "default_example": 0},
    "earliest_cr_line": {"type": "numeric", "question": "📆 What year did you open your very first credit line?", "default_example": 2005},
    "fico_range_high": {"type": "numeric", "question": "🔢 What's the higher number in your FICO score range?", "default_example": 724},
    "fico_range_low": {"type": "numeric", "question": "🔢 And the lower number in your FICO score range?", "default_example": 720},
    "inq_last_6mths": {"type": "numeric", "question": "📑 How many credit inquiries in the past 6 months? (Excluding auto/mortgage)", "default_example": 0},
    "open_acc": {"type": "numeric", "question": "📂 How many open credit accounts (cards, loans) do you have?", "default_example": 8},
    "pub_rec": {"type": "numeric", "question": "📢 Any public records like bankruptcies?", "default_example": 0},
    "revol_bal": {"type": "numeric", "question": "💸 What's your total balance on revolving accounts (like credit cards)?", "default_example": 15000},
    "revol_util": {"type": "numeric", "question": "📉 What's your revolving credit utilization rate? (Balance / Limit, %)", "default_example": 45.5},
    "total_acc": {"type": "numeric", "question": "📁 In total, how many credit accounts have you ever had (open or closed)?", "default_example": 20},
    "mort_acc": {"type": "numeric", "question": "🏦 How many mortgage accounts do you have?", "default_example": 1},

    # --- Categorical Features ---
    "term": {"type": "categorical", "question": "📅 And for how long? (Loan term)", "options": ["36 months", "60 months"], "default_example": "36 months"},
    "grade": {"type": "categorical", "question": "🔤 What's the loan grade assigned?", "options": ["A", "B", "C", "D", "E", "F", "G"], "default_example": "B"},
    "sub_grade": {"type": "categorical", "question": "🔠 And the sub-grade?", "options": sorted([f"{g}{i}" for g in ["A","B","C","D","E","F","G"] for i in range(1, 6)]), "default_example": "B3"},
    "emp_length": {"type": "categorical", "question": "👷 How long have you been with your current employer?",
                   "options": ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years", "n/a"], "default_example": "10+ years"},
    "home_ownership": {"type": "categorical", "question": "🏠 What's your home situation?", "options": ["RENT", "MORTGAGE", "OWN", "ANY", "OTHER", "NONE"], "default_example": "MORTGAGE"},
    "verification_status": {"type": "categorical", "question": "🔍 Has your income been verified?", "options": ["Verified", "Source Verified", "Not Verified"], "default_example": "Source Verified"},
    "purpose": {"type": "categorical", "question": "🎯 What's the main reason for this loan?",
                "options": ["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase", "small_business", "car", "medical", "moving", "vacation", "house", "wedding", "renewable_energy", "educational"], "default_example": "debt_consolidation"},
    "addr_state": {"type": "categorical", "question": "🗺️ Which state do you live in?",
                   "options": ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"], "default_example": "CA"},
    "application_type": {"type": "categorical", "question": "🧑‍🤝‍🧑 Is this just for you, or a joint application?", "options": ["Individual", "Joint App"], "default_example": "Individual"},
}


# 2. Determine which conceptual features to ask about
# (Keep logic as before)
conceptual_features_to_ask = []
processed_model_features = set(model_features)
for conceptual_name, info in conceptual_features_info.items():
    is_needed = False
    if info["type"] == "numeric":
        if conceptual_name in processed_model_features: is_needed = True
        elif conceptual_name == 'earliest_cr_line' and any(f.startswith('credit_history') for f in model_features): is_needed = True
    elif info["type"] == "categorical":
        if any(f.startswith(conceptual_name + "_") for f in model_features): is_needed = True
        elif conceptual_name in processed_model_features: is_needed = True
    if is_needed: conceptual_features_to_ask.append(conceptual_name)

if not conceptual_features_to_ask:
    st.error("Error: Could not determine which features to ask the user.")
    st.stop()

# --- Session State Initialization ---
if "messages" not in st.session_state: st.session_state.messages = []
if "user_inputs" not in st.session_state: st.session_state.user_inputs = {}
if "current_feature_index" not in st.session_state: st.session_state.current_feature_index = 0
if "prediction_done" not in st.session_state: st.session_state.prediction_done = False
if "processing_input" not in st.session_state: st.session_state.processing_input = False

# --- Helper Functions ---

def get_current_feature_to_ask():
    """Gets the name of the conceptual feature we need input for next."""
    idx = st.session_state.current_feature_index
    if 0 <= idx < len(conceptual_features_to_ask):
        feature_name = conceptual_features_to_ask[idx]
        if feature_name not in st.session_state.user_inputs:
            return feature_name
    return None

def add_assistant_message(content, feature_name=None, requires_input=False):
    """Adds a message from the assistant, potentially including default/example."""
    full_content = content
    # Append default example to the question text if input is required
    if requires_input and feature_name and feature_name in conceptual_features_info:
        example = conceptual_features_info[feature_name].get("default_example")
        if example is not None:
            full_content = f"{content} (e.g., {example})" # Add example here

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_content,
        "feature_name": feature_name,
        "requires_input": requires_input
    })

def add_user_message(content):
    """Adds a message from the user to the history."""
    st.session_state.messages.append({"role": "user", "content": content})

# --- Preprocessing and Prediction Functions ---
# (Keep preprocess_input and make_prediction/run_prediction_logic as defined in the previous response,
#  ensure they handle potential errors and update state correctly)
def preprocess_input(raw_input_dict):
    """Applies preprocessing to match the model's expected features."""
    errors = []
    input_df = pd.DataFrame([raw_input_dict])
    asked_categorical = [f for f in raw_input_dict.keys() if f in conceptual_features_info and conceptual_features_info[f]['type'] == 'categorical']
    asked_numeric = [f for f in raw_input_dict.keys() if f in conceptual_features_info and conceptual_features_info[f]['type'] == 'numeric']

    # 1. Type Conversion
    for col in asked_numeric:
        if col in input_df.columns:
            original_value = input_df[col].iloc[0]
            try:
                if original_value is None or str(original_value).strip() == "":
                    input_df[col] = np.nan
                else:
                    cleaned_value = str(original_value).replace('%', '').strip()
                    input_df[col] = pd.to_numeric(cleaned_value)
            except (ValueError, TypeError):
                errors.append(f"Hmm, '{original_value}' for '{col}' doesn't look like a number. Treating as missing for now.")
                input_df[col] = np.nan

    # 2. Feature Engineering
    if 'earliest_cr_line' in input_df.columns and any(f.startswith('credit_history') for f in model_features):
         try:
            cr_line_year = pd.to_numeric(input_df['earliest_cr_line'].iloc[0])
            if pd.isna(cr_line_year) or cr_line_year < 1900 or cr_line_year > pd.Timestamp.now().year:
                 errors.append(f"That year for 'earliest_cr_line' seems unlikely. Calculation skipped.")
                 input_df['credit_history_length'] = np.nan
            else:
                 current_year = pd.Timestamp.now().year
                 input_df['credit_history_length'] = current_year - cr_line_year
            if 'earliest_cr_line' not in model_features:
                 input_df = input_df.drop(columns=['earliest_cr_line'])
         except Exception:
             errors.append(f"Couldn't calculate credit history length from the year provided.")
             if 'credit_history_length' in model_features: input_df['credit_history_length'] = np.nan
             if 'earliest_cr_line' not in model_features and 'earliest_cr_line' in input_df.columns: input_df = input_df.drop(columns=['earliest_cr_line'])

    # 3. Encoding
    if asked_categorical:
        try:
            for col in asked_categorical:
                if col in input_df.columns: input_df[col] = input_df[col].astype(str)
            input_df_processed = pd.get_dummies(input_df, columns=asked_categorical, dummy_na=False)
        except Exception as e:
            errors.append(f"Had trouble encoding the categories: {e}")
            return None, errors
    else: input_df_processed = input_df.copy()

    # 4. Align columns
    try:
        missing_cols = set(model_features) - set(input_df_processed.columns)
        for c in missing_cols: input_df_processed[c] = 0
        input_df_processed = input_df_processed[model_features]
    except Exception as e:
        errors.append(f"Couldn't quite align the data for the model: {e}")
        return None, errors

    # 5. Handle Missing Values
    if input_df_processed.isnull().values.any():
        nan_cols = input_df_processed.columns[input_df_processed.isnull().any()].tolist()
        errors.append(f"Looks like some info was missing ({', '.join(nan_cols)}). Filled with 0 for now.")
        input_df_processed = input_df_processed.fillna(0)
    return input_df_processed, errors


def make_prediction():
    """Starts the prediction process by showing a message."""
    st.session_state.processing_input = True
    add_assistant_message("Okay, crunching the numbers now... 🤔")
    st.rerun()

def run_prediction_logic():
    """Runs the actual preprocessing and prediction steps."""
    time.sleep(1) # Simulate thinking
    raw_inputs = st.session_state.user_inputs
    processed_df, preprocessing_errors = preprocess_input(raw_inputs.copy())

    # Add preprocessing errors to chat if any
    if preprocessing_errors:
        error_message = "⚠️ **Whoops, had some trouble processing the info:**\n\n*   " + "\n*   ".join(preprocessing_errors)
        # Avoid adding duplicate error messages if already present
        if not st.session_state.messages or st.session_state.messages[-1].get("content") != error_message:
            add_assistant_message(error_message)
        # Stop if critical errors occurred
        if any("Error:" in e for e in preprocessing_errors) or processed_df is None:
             st.session_state.prediction_done = True
             # Avoid adding duplicate "can't predict" messages
             fail_msg = "😥 Couldn't make a prediction due to those issues. Maybe we should restart?"
             if not st.session_state.messages or st.session_state.messages[-1].get("content") != fail_msg:
                 add_assistant_message(fail_msg)
             st.session_state.processing_input = False # Allow restart
             return

    # Proceed with prediction if data is okay
    if processed_df is not None:
        try:
            processed_df = processed_df.astype(float)
            prediction = model.predict(processed_df)[0]
            probability = model.predict_proba(processed_df)[0]
            result_message = "**Prediction Result:**\n\n"
            if prediction == 1:
                result_message += f"✅ Good news! Looks like this loan is likely to be **Approved / Fully Paid**. (Confidence: {probability[1]:.1%})"
                st.balloons()
            else:
                result_message += f"❌ Hmm, based on the info, this loan seems more likely to be **Rejected / Default**. (Confidence: {probability[0]:.1%})"
            add_assistant_message(result_message)
            st.session_state.prediction_done = True
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            fail_msg = f"😥 Yikes! Something unexpected happened during the final prediction step. Error: {e}"
            if not st.session_state.messages or st.session_state.messages[-1].get("content") != fail_msg:
                add_assistant_message(fail_msg)
            st.session_state.prediction_done = True
    else:
         # Fallback if processed_df is None without critical errors logged (less likely now)
         fail_msg = "😥 Couldn't proceed with prediction due to input processing issues."
         if not st.session_state.messages or st.session_state.messages[-1].get("content") != fail_msg:
             add_assistant_message(fail_msg)
         st.session_state.prediction_done = True

    st.session_state.processing_input = False # Prediction finished


def restart_conversation():
    """Resets the session state."""
    st.session_state.messages = []
    st.session_state.user_inputs = {}
    st.session_state.current_feature_index = 0
    st.session_state.prediction_done = False
    st.session_state.processing_input = False
    add_assistant_message("👋 Hi there! I'm Loan Pal, ready to help predict your loan status.")
    add_assistant_message("I'll ask a few questions. Let's get started!")
    time.sleep(0.5)
    first_feature = get_current_feature_to_ask()
    if first_feature:
        info = conceptual_features_info[first_feature]
        # Pass the raw question here; add_assistant_message will append the example
        add_assistant_message(info["question"], feature_name=first_feature, requires_input=True)
    else:
        add_assistant_message("Hmm, looks like I don't have any questions to ask right now.")

# --- Main Chat Interface Logic ---

st.title("💬 Loan Pal Chatbot")
st.caption("Your friendly loan status predictor")
st.markdown("---")

if not st.session_state.messages:
    restart_conversation()

# Display chat history and input widgets
chat_placeholder = st.container()
with chat_placeholder:
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"]) # Displays question with example now

            # Logic to display input widget for the *current* question
            current_feature_asking = message.get("feature_name")
            if (message.get("requires_input") and
                current_feature_asking and
                current_feature_asking not in st.session_state.user_inputs and
                not st.session_state.prediction_done and
                not st.session_state.processing_input):

                info = conceptual_features_info[current_feature_asking]
                widget_key = f"input_{current_feature_asking}"
                user_answer = None
                default_example = info.get("default_example") # Get default for placeholder

                if info["type"] == "categorical":
                    options = info.get("options", [])
                    options_with_prompt = ["-- Select --"] + options
                    user_answer = st.selectbox(
                        "Your selection:", options=options_with_prompt, key=widget_key,
                        index=0, label_visibility="collapsed"
                    )
                    if user_answer != "-- Select --":
                         st.session_state.processing_input = True

                elif info["type"] == "numeric":
                    # Use placeholder for the example/format suggestion
                    placeholder_text = f"e.g., {default_example}" if default_example is not None else "Enter number..."
                    user_answer = st.number_input(
                        "Your answer:", key=widget_key, value=None,
                        placeholder=placeholder_text, # Use placeholder here
                        step=1.0 if isinstance(default_example, int) or default_example == 0 else 0.01, # Basic step logic
                        format=None, label_visibility="collapsed"
                    )
                    if user_answer is not None: # Trigger on any number input
                         st.session_state.processing_input = True

                # --- Centralized Input Processing ---
                if st.session_state.processing_input and user_answer is not None and user_answer != "-- Select --":
                    st.session_state.user_inputs[current_feature_asking] = user_answer
                    add_user_message(str(user_answer))
                    st.session_state.current_feature_index += 1
                    next_feature = get_current_feature_to_ask()
                    if next_feature:
                         next_info = conceptual_features_info[next_feature]
                         # Pass raw question; example added by the function
                         add_assistant_message(f"Got it! Next: {next_info['question']}", feature_name=next_feature, requires_input=True)
                         st.session_state.processing_input = False
                         st.rerun()
                    else:
                         make_prediction() # Starts prediction process


# --- Handle the actual prediction logic if triggered ---
if st.session_state.get("processing_input") and st.session_state.prediction_done is False and get_current_feature_to_ask() is None:
    if st.session_state.messages and "crunching the numbers" in st.session_state.messages[-1].get("content", ""):
        run_prediction_logic()
        st.rerun()

# --- Restart Button ---
st.markdown("---")
# Place button at the bottom more subtly
if st.button("🔄 Start Over", key="restart_button"):
        restart_conversation()
        st.rerun()