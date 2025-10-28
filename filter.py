import pandas as pd


df_le  = pd.read_csv("Life Expectancy Data.csv")
df_co2 = pd.read_csv("co2_emission.csv")
df_gdp = pd.read_csv("f5a9ad86-f7cb-42ba-868e-b444c1d52fa4_Data.csv")


countries = ["Argentina","Australia","Brazil","Canada","China","France","Germany","Italy",
             "India","Indonesia","Japan","Korea, Rep.","Mexico","Russia","Russian Federation",
             "Saudi Arabia","South Africa","United Kingdom","United States"]
years = list(range(2000, 2016))


df_co2 = df_co2[["Entity", "Year", "Annual CO₂ emissions (tonnes )"]].rename(
    columns={"Entity": "Country", "Annual CO₂ emissions (tonnes )": "CO2 Use"}
)
df_co2 = df_co2[df_co2["Country"].isin(countries) & df_co2["Year"].isin(years)]

df_gdp = df_gdp[["Country Name", "Time", "Value"]].rename(
    columns={"Country Name": "Country", "Time": "Year", "Value": "GDP per capita USD"}
)
df_gdp = df_gdp[df_gdp["Country"].isin(countries) & df_gdp["Year"].isin(years)]

df_le = df_le[["Country", "Year", "Life expectancy "]].rename(
    columns={"Life expectancy ": "Life Expectancy"}
)
df_le = df_le[df_le["Country"].isin(countries) & df_le["Year"].isin(years)]


dataset_A = pd.merge(df_gdp, df_le, on=["Country", "Year"], how="inner")[
    ["Country", "Year", "GDP per capita USD", "Life Expectancy"]
]
dataset_A.to_csv("dataset_A_gdp_life.csv", index=False)


dataset_B = pd.merge(df_gdp, df_co2, on=["Country", "Year"], how="inner")
dataset_B = pd.merge(dataset_B, df_le, on=["Country", "Year"], how="inner")[
    ["Country", "Year", "GDP per capita USD", "CO2 Use", "Life Expectancy"]
]
dataset_B.to_csv("dataset_B_gdp_co2_life.csv", index=False)

dataset_C = dataset_B.copy()
dataset_C.to_csv("dataset_C_gdp_co2_year_life.csv", index=False)

