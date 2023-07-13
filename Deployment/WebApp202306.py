import pandas as pd
import joblib
import streamlit as st


@st.cache_resource
def load_model():
    return joblib.load("../output/ModelMasterDict202306.pkl")


@st.cache_resource
def load_default():
    return joblib.load("../output/default_values.pkl")


@st.cache_data
def load_value_range():
    return pd.read_csv("../output/feature_limits.csv")


@st.cache_data
def predict(segment, _ModelDict, df):
    model = _ModelDict["model"][segment]
    feature = _ModelDict["feature"][segment]

    # df["Adoption_Pred"] = model.predict(df[feature])
    Adoption_Pred = model.predict(df[feature])

    return Adoption_Pred


def main():
    
    ModelDict = load_model()

    default_values = load_default()

    limit_all = load_value_range()

    st.title("EV Adoption Model v1")

    # header = st.container()
    # header.write("Here is a sticky header")
    # header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)

    # ### Custom CSS for the sticky header
    # st.markdown(
    #     """
    # <style>
    #     div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
    #         position: sticky;
    #         top: 2.875rem;
    #         background-color: white;
    #         z-index: 999;
    #     }
    #     .fixed-header {
    #         border-bottom: 1px solid black;
    #     }
    # </style>
    #     """,
    #     unsafe_allow_html=True
    # )

    defaults = {
        "availability": 0,
        "Prob_VeryLikely": 0,
        "AffordScore_MA3": 0,
        "RatioGasChargingCost_MA3": 0,
        "ChargingPortsPerCapita": 0,
        "Infr_avail": 0,
        "Infr_location": 0,
        "Infr_time": 0,
        "EV_Count": 0,
        "DaysSupply_EV_ICE_Ratio_c": 0,
        "DaysToTurn_EV_ICE_Ratio": 0,
    }

    limits = {
        "availability": (0, 0),
        "Prob_VeryLikely": (0, 0),
        "AffordScore_MA3": (0, 0),
        "RatioGasChargingCost_MA3": (0, 0),
        "ChargingPortsPerCapita": (0, 0),
        "Infr_avail": (0, 0),
        "Infr_location": (0, 0),
        "Infr_time": (0, 0),
        "EV_Count": (0, 0),
        "DaysSupply_EV_ICE_Ratio_c": (0, 0),
        "DaysToTurn_EV_ICE_Ratio": (0, 0),
    }

    col1, col2, col3 = st.columns(3)

    with col1:
        segment = st.selectbox(
            "PIN Segment:",
            [
                "Compact SUV",
                "Compact Car",
                "Large Premium Car",
                "Small Premium Car",
                "Compact Sporty Car",
                "Small SUV",
                "Large Pickup - LD",
                "Large Van",
                "Small Premium SUV",
            ],
        )

        df_limits = limit_all.loc[limit_all["segment"] == segment]
        for f in df_limits["feature"].values:
            defaults[f] = default_values[f][segment]
            limits[f] = df_limits[df_limits["feature"] == f][["lower", "upper"]].values[0]
            if defaults[f] < limits[f][0]:
                defaults[f] = limits[f][0]
            elif defaults[f] > limits[f][1]:
                defaults[f] = limits[f][1]

        # avail_default = default_values["availability"][segment]
        # Interest_default = default_values["Prob_VeryLikely"][segment]
        # afford_default = default_values["AffordScore_MA3"][segment]
        # GasCharging_default = default_values["RatioGasChargingCost_MA3"][segment]
        # Ports_default = default_values["ChargingPortsPerCapita"][segment]
        # Infr_avail_default = default_values["Infr_avail"][segment]
        # Infr_location_default = default_values["Infr_location"][segment]
        # Infr_time_default = default_values["Infr_time"][segment]
        # EV_Count_default = default_values["EV_Count"][segment]
        # DaysSupply_default = default_values["DaysSupply_EV_ICE_Ratio_c"][segment]
        # DaysToTurn_default = default_values["DaysToTurn_EV_ICE_Ratio"][segment]

        availability = st.number_input(
            "Availability Score",
            min_value=limits["availability"][0],
            max_value=limits["availability"][1],
            value=defaults["availability"],
        )

        # CARBFlag = st.selectbox("CARB State", ["No", "Yes"])

        Prob_VeryLikely = st.number_input(
            "Interest (between 0 and 1)",
            min_value=limits["Prob_VeryLikely"][0],
            max_value=limits["Prob_VeryLikely"][1],
            value=defaults["Prob_VeryLikely"],
        )

        AffordScore_MA3 = st.number_input(
            "Affordability Score (MA3)",
            min_value=limits["AffordScore_MA3"][0],
            max_value=limits["AffordScore_MA3"][1],
            value=defaults["AffordScore_MA3"],
        )

        RatioGasChargingCost_MA3 = st.number_input(
            "Gas Charging Cost Ratio",
            min_value=limits["RatioGasChargingCost_MA3"][0],
            max_value=limits["RatioGasChargingCost_MA3"][1],
            value=defaults["RatioGasChargingCost_MA3"],
        )

        ChargingPortsPerCapita = st.number_input(
            "Charging Ports Per Capita",
            format="%.6f",
            step=1e-6,
            min_value=limits["ChargingPortsPerCapita"][0],
            max_value=limits["ChargingPortsPerCapita"][1],
            value=defaults["ChargingPortsPerCapita"],
        )

        Infr_avail = st.number_input(
            "Infrastructure-Availability",
            min_value=limits["Infr_avail"][0],
            max_value=limits["Infr_avail"][1],
            value=defaults["Infr_avail"],
        )

    with col2:
        Infr_location = st.number_input(
            "Infrastructure-Location",
            min_value=limits["Infr_location"][0],
            max_value=limits["Infr_location"][1],
            value=defaults["Infr_location"],
        )

        Infr_time = st.number_input(
            "Infrastructure-Time",
            min_value=limits["Infr_time"][0],
            max_value=limits["Infr_time"][1],
            value=defaults["Infr_time"],
        )

        EV_Count = st.number_input(
            "EV Count",
            min_value=int(limits["EV_Count"][0]),
            max_value=int(limits["EV_Count"][1]),
            value=int(defaults["EV_Count"]),
        )

        DaysSupply_EV_ICE_Ratio_c = st.number_input(
            "EV ICE Days Supply Ratio",
            min_value=limits["DaysSupply_EV_ICE_Ratio_c"][0],
            max_value=limits["DaysSupply_EV_ICE_Ratio_c"][1],
            value=defaults["DaysSupply_EV_ICE_Ratio_c"],
        )

        DaysToTurn_EV_ICE_Ratio = st.number_input(
            "EV ICE Days to Turn Ratio",
            min_value=limits["DaysToTurn_EV_ICE_Ratio"][0],
            max_value=limits["DaysToTurn_EV_ICE_Ratio"][1],
            value=defaults["DaysToTurn_EV_ICE_Ratio"],
        )

        NoEV_ind = st.selectbox("No EV Inventory", ["No", "Yes"])

    data = {
        "Age_Under18Years_Pop_Pct": 0.22454363119977208,
        "Age_55YearsAndOlder_Pop_Pct": 0.2894230317372937,
        "EducationLevel_GraduateOrProfessionalDegree_Pop_Pct": 0.12527361037048007,
        "WorkCommute_AvgMinutesToWork_Pop_Cnt": 26.942917436806297,
        "Income_MedianHouseholdIncome_Hh_Cnt": 70668.1420719971,
        "EducationLevel_BachelorsDegree_Pop_Pct": 0.20031474625806445,
        "Age_25To34Years_Pop_Pct": 0.13907340061972506,
        "PctDemocrat": 0.5118391872017257,
        "WorkCommute_PublicTransportation_Pop_Pct": 0.044554294730160884,
        "Prob_VeryLikely": 0.26112004321067656,
        "Age_45To54Years_Pop_Pct": 0.12724744763584744,
        "Age_35To44Years_Pop_Pct": 0.1265554836602792,
        "NatWalkInd": 8.365970611965075,
        "tmin": -5.250546670121218,
    }

    data["availability"] = availability
    data["availability"] = availability
    # data["CARBFlag"] = int(CARBFlag == "Yes")
    data["CARBFlag"] = 0
    data["Prob_VeryLikely"] = Prob_VeryLikely
    data["AffordScore_MA3"] = AffordScore_MA3
    data["RatioGasChargingCost_MA3"] = RatioGasChargingCost_MA3
    data["ChargingPortsPerCapita"] = ChargingPortsPerCapita
    data["Infr_avail"] = Infr_avail
    data["Infr_location"] = Infr_location
    data["Infr_time"] = Infr_time
    data["EV_Count"] = EV_Count
    data["DaysSupply_EV_ICE_Ratio_c"] = DaysSupply_EV_ICE_Ratio_c
    data["DaysToTurn_EV_ICE_Ratio"] = DaysToTurn_EV_ICE_Ratio
    data["NoEV_ind"] = int(NoEV_ind == "Yes")
    data["TeslaQtrBegin_ind"] = 0

    df = pd.DataFrame(data, index=[0])

    Adoption_Pred = predict(segment, ModelDict, df)

    with col3:
        st.write(
            "<h4 style='text-align: center;'>Adoption Estimate: {}</h4>".format(round(Adoption_Pred[0], 2)),
            unsafe_allow_html=True,
        )

        st.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)

        ### Custom CSS for the sticky header
        st.markdown(
            """
            <style>
                div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
                    position: sticky;
                    top: 2.875rem;
                    background-color: white;
                    z-index: 999;
                }
                .fixed-header {
                    border-bottom: 1px solid black;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # css='''
    #     <style>
    #         section.main>div {
    #             padding-bottom: 1rem;
    #         }
    #         [data-testid="column"]>div>div>div>div>div {
    #             overflow: auto;
    #             height: 70vh;
    #         }
    #     </style>
    #     '''

    # st.markdown(css, unsafe_allow_html=True)

    # Add CSS styling for scrolling

    # from streamlit.components.v1 import html

    # button = """<script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="blackarysf" data-color="#FFDD00" data-emoji=""  data-font="Cookie" data-text="Adoption Estimate" data-outline-color="#000000" data-font-color="#000000" data-coffee-color="#ffffff" ></script>"""

    # html(button, height=70, width=220)

    # st.markdown(
    #     """
    #     <style>
    #         iframe[width="220"] {
    #             position: fixed;
    #             bottom: 600px;
    #             right: 40px;
    #         }
    #     </style>
    #     """,
    #     unsafe_allow_html=True,
    # )

    st.write(
        '<style>body { margin: 0; font-family: Arial, Helvetica, sans-serif;} .header{padding: 10px 16px; background: #555; color: #f1f1f1; position:fixed;top:0;} .sticky { position: fixed; top: 10px; width: 100%;} </style><div class="header" id="myHeader">'
        + str(Adoption_Pred[0])
        + "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
