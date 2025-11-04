# predict_for_user.py
# This script loads the trained models and runs the prediction pipeline.

import os, joblib, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# --- CONFIG ---
MODEL_DIR = "models_user"
# These lists MUST match the features your models were trained on
NR_FEATURES = ['temperature', 'rainfall', 'soil_moisture', 'coating_thickness', 'soil_pH']
FR_FEATURES = ['crop_enc', 'soil_pH', 'N', 'P', 'K', 'nutrient_release_rate']

# ------------------------------------------------
# 1. LOAD ALL MODELS AND ENCODERS
# ------------------------------------------------
print("Loading models... Please wait.")

try:
    # Module 1 Models
    model_nr_rate = joblib.load(os.path.join(MODEL_DIR, "nutrient_release_model.joblib"))
    model_nr_cat = joblib.load(os.path.join(MODEL_DIR, "nutrient_release_category_clf.joblib"))
    le_nr_cat = joblib.load(os.path.join(MODEL_DIR, "le_nr_category.joblib"))

    # Module 2 Models
    model_fert_type = joblib.load(os.path.join(MODEL_DIR, "fert_type_clf.joblib"))
    model_fert_amount = joblib.load(os.path.join(MODEL_DIR, "fert_amount_reg.joblib"))
    model_fert_timing = joblib.load(os.path.join(MODEL_DIR, "fert_timing_reg.joblib"))
    
    # Encoders
    le_crop = joblib.load(os.path.join(MODEL_DIR, "le_crop.joblib"))
    le_fert = joblib.load(os.path.join(MODEL_DIR, "le_fert.joblib"))

    print("✅ All models loaded successfully.")

except FileNotFoundError:
    print(f"[ERROR] Models not found in directory '{MODEL_DIR}'.")
    print("Please run 'train_user_models.py' first to generate the models.")
    exit()
except Exception as e:
    print(f"An error occurred while loading models: {e}")
    exit()

# ------------------------------------------------
# 2. USER INPUT & PREDICTION FUNCTION
# ------------------------------------------------

def predict_for_user():
    print("\n==============================================")
    print("🌾 Enter New Farm and Environmental Details 🌾")
    print("==============================================")
    
    try:
        # --- Get Inputs for Module 1 ---
        print("\n[Module 1: Environmental & Fertilizer Details]")
        temp = float(input("  Temperature (°C): "))
        rain = float(input("  Rainfall (mm/day): "))
        moisture = float(input("  Soil Moisture (0.1–0.5): "))
        coating = float(input("  Coating Thickness (mm): "))
        soil_pH = float(input("  Soil pH: "))

        # --- Get Inputs for Module 2 ---
        print("\n[Module 2: Crop & Soil Details]")
        crop_in = input(f"  Crop Type (e.g., {', '.join(le_crop.classes_)}): ").strip()
        N = float(input("  Nitrogen (N) in soil: "))
        P = float(input("  Phosphorus (P) in soil: "))
        K = float(input("  Potassium (K) in soil: "))

        # ---------------------------------
        # --- STEP 1: MODULE 1 PREDICTION ---
        # ---------------------------------
        
        # Create input DataFrame for Module 1
        df_input_nr = pd.DataFrame([{
            'temperature': temp,
            'rainfall': rain,
            'soil_moisture': moisture,
            'coating_thickness': coating,
            'soil_pH': soil_pH
        }])
        df_input_nr = df_input_nr[NR_FEATURES] # Ensure column order

        # Predict rate and category
        pred_release_rate = model_nr_rate.predict(df_input_nr)[0]
        
        # FIX: Add a "floor" at 0. A release rate cannot be negative.
        pred_release_rate = max(0, pred_release_rate)
        
        pred_release_cat_enc = model_nr_cat.predict(df_input_nr)[0]
        pred_release_category = le_nr_cat.inverse_transform([pred_release_cat_enc])[0]

        # ---------------------------------
        # --- STEP 2: MODULE 2 PREDICTION ---
        # ---------------------------------

        # Encode user's crop input
        try:
            crop_enc = le_crop.transform([crop_in])[0]
        except:
            print(f"Warning: Crop '{crop_in}' not in model. Using default '{le_crop.classes_[0]}'.")
            crop_enc = le_crop.transform([le_crop.classes_[0]])[0]

        # Create input DataFrame for Module 2
        df_input_fr = pd.DataFrame([{
            'crop_enc': crop_enc,
            'soil_pH': soil_pH,
            'N': N,
            'P': P,
            'K': K,
            'nutrient_release_rate': pred_release_rate  # The KEY link
        }])
        df_input_fr = df_input_fr[FR_FEATURES] # Ensure column order
        
        # Predict type, amount, and timing
        pred_type_enc = model_fert_type.predict(df_input_fr)[0]
        pred_amount = model_fert_amount.predict(df_input_fr)[0]
        pred_timing = model_fert_timing.predict(df_input_fr)[0]

        # Decode the fertilizer type
        pred_type = le_fert.inverse_transform([pred_type_enc])[0]

        # ---------------------------------
        # --- STEP 3: DISPLAY RESULTS ---
        # ---------------------------------
        
        print("\n-------------------------------------------")
        print("💡 SMART FERTILIZER RECOMMENDATION 💡")
        print("-------------------------------------------")
        
        print("\n--- Nutrient Release Prediction (Module 1) ---")
        print(f"  Predicted Release Rate:     {pred_release_rate:.2f} (% per day)")
        print(f"  Predicted Release Category: {pred_release_category.capitalize()}")
        
        print("\n--- Fertilizer Recommendation (Module 2) ---")
        print(f"  Recommended Fertilizer Type:  {pred_type}")
        print(f"  Recommended Application Amount: {pred_amount:.2f} kg/ha")
        print(f"  Recommended Application Timing: {pred_timing:.1f} days after sowing")
        print("-------------------------------------------")

    except ValueError:
        print("\n[Error] Invalid input. Please enter numerical values only.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

# ------------------------------------------------
# 3. RUN THE APPLICATION
# ------------------------------------------------

if __name__ == "__main__":
    predict_for_user()
