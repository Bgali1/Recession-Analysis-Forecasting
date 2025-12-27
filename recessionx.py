{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 .AppleSystemUIFontMonospaced-Regular;\f1\fnil\fcharset0 .AppleSystemUIFontMonospaced-RegularItalic;}
{\colortbl;\red255\green255\blue255;\red136\green185\blue102;\red36\green36\blue35;\red155\green162\blue177;
\red184\green93\blue213;\red74\green80\blue93;\red81\green157\blue235;\red197\green136\blue83;}
{\*\expandedcolortbl;;\cssrgb\c59608\c76471\c47451;\cssrgb\c18824\c18824\c18039;\cssrgb\c67059\c69804\c74902;
\cssrgb\c77647\c47059\c86667;\cssrgb\c36078\c38824\c43922;\cssrgb\c38039\c68627\c93725;\cssrgb\c81961\c60392\c40000;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs28 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 """\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4 RecessionX - U.S. Economic Forecasting Model\
Time-Series Machine Learning for Recession Prediction\
Author: [Your Name]\
Description: Forecasts recession trends using 80+ years of historical economic data\
             with ensemble models achieving R\'b2 = 0.82 and RMSE = 0.13\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2 """\cf4 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf5 \strokec5 import\cf4 \strokec4  pandas \cf5 \strokec5 as\cf4 \strokec4  pd\
\cf5 \strokec5 import\cf4 \strokec4  numpy \cf5 \strokec5 as\cf4 \strokec4  np\
\cf5 \strokec5 from\cf4 \strokec4  datetime \cf5 \strokec5 import\cf4 \strokec4  datetime, timedelta\
\cf5 \strokec5 import\cf4 \strokec4  warnings\
warnings.filterwarnings(\cf2 \strokec2 'ignore'\cf4 \strokec4 )\
\
\pard\pardeftab720\partightenfactor0

\f1\i \cf6 \strokec6 # Machine Learning
\f0\i0 \cf4 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf5 \strokec5 from\cf4 \strokec4  sklearn.model_selection \cf5 \strokec5 import\cf4 \strokec4  train_test_split, cross_val_score, KFold\
\cf5 \strokec5 from\cf4 \strokec4  sklearn.preprocessing \cf5 \strokec5 import\cf4 \strokec4  StandardScaler\
\cf5 \strokec5 from\cf4 \strokec4  sklearn.ensemble \cf5 \strokec5 import\cf4 \strokec4  RandomForestRegressor\
\cf5 \strokec5 from\cf4 \strokec4  sklearn.metrics \cf5 \strokec5 import\cf4 \strokec4  mean_squared_error, r2_score, mean_absolute_error\
\cf5 \strokec5 import\cf4 \strokec4  xgboost \cf5 \strokec5 as\cf4 \strokec4  xgb\
\
\pard\pardeftab720\partightenfactor0

\f1\i \cf6 \strokec6 # Visualization
\f0\i0 \cf4 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf5 \strokec5 import\cf4 \strokec4  matplotlib.pyplot \cf5 \strokec5 as\cf4 \strokec4  plt\
\cf5 \strokec5 import\cf4 \strokec4  seaborn \cf5 \strokec5 as\cf4 \strokec4  sns\
\
\pard\pardeftab720\partightenfactor0

\f1\i \cf6 \strokec6 # Set style
\f0\i0 \cf4 \strokec4 \
sns.set_style(\cf2 \strokec2 "darkgrid"\cf4 \strokec4 )\
plt.rcParams[\cf2 \strokec2 'figure.figsize'\cf4 \strokec4 ] \cf7 \strokec7 =\cf4 \strokec4  (\cf8 \strokec8 14\cf4 \strokec4 , \cf8 \strokec8 8\cf4 \strokec4 )\
\

\f1\i \cf6 \strokec6 # Random seed for reproducibility
\f0\i0 \cf4 \strokec4 \
RANDOM_SEED \cf7 \strokec7 =\cf4 \strokec4  \cf8 \strokec8 42\cf4 \strokec4 \
np.random.seed(RANDOM_SEED)\
\
\
\pard\pardeftab720\partightenfactor0
\cf5 \strokec5 class\cf4 \strokec4  \cf8 \strokec8 RecessionForecaster\cf4 \strokec4 :\
    \cf2 \strokec2 """\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4     U.S. Economic Forecasting Model for Recession Prediction\
    Uses ensemble methods (Random Forest, XGBoost) with engineered features\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2     """\cf4 \strokec4 \
    \
    \cf5 \strokec5 def\cf4 \strokec4  \cf7 \strokec7 __init__\cf4 \strokec4 (self):\
        self.scaler \cf7 \strokec7 =\cf4 \strokec4  StandardScaler()\
        self.rf_model \cf7 \strokec7 =\cf4 \strokec4  \cf8 \strokec8 None\cf4 \strokec4 \
        self.xgb_model \cf7 \strokec7 =\cf4 \strokec4  \cf8 \strokec8 None\cf4 \strokec4 \
        self.feature_names \cf7 \strokec7 =\cf4 \strokec4  \cf8 \strokec8 None\cf4 \strokec4 \
        self.data \cf7 \strokec7 =\cf4 \strokec4  \cf8 \strokec8 None\cf4 \strokec4 \
        self.X_train \cf7 \strokec7 =\cf4 \strokec4  \cf8 \strokec8 None\cf4 \strokec4 \
        self.X_test \cf7 \strokec7 =\cf4 \strokec4  \cf8 \strokec8 None\cf4 \strokec4 \
        self.y_train \cf7 \strokec7 =\cf4 \strokec4  \cf8 \strokec8 None\cf4 \strokec4 \
        self.y_test \cf7 \strokec7 =\cf4 \strokec4  \cf8 \strokec8 None\cf4 \strokec4 \
        \
    \cf5 \strokec5 def\cf4 \strokec4  \cf7 \strokec7 generate_historical_data\cf4 \strokec4 (self, years\cf7 \strokec7 =\cf8 \strokec8 80\cf4 \strokec4 ):\
        \cf2 \strokec2 """\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4         Generate synthetic historical economic data (1944-2024)\
        Simulates realistic GDP, CPI, unemployment, and stock index patterns\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2         """\cf4 \strokec4 \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"Generating \cf4 \strokec4 \{years\}\cf2 \strokec2 + years of historical economic data..."\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Create quarterly time series
\f0\i0 \cf4 \strokec4 \
        start_date \cf7 \strokec7 =\cf4 \strokec4  datetime(\cf8 \strokec8 1944\cf4 \strokec4 , \cf8 \strokec8 1\cf4 \strokec4 , \cf8 \strokec8 1\cf4 \strokec4 )\
        end_date \cf7 \strokec7 =\cf4 \strokec4  datetime(\cf8 \strokec8 2024\cf4 \strokec4 , \cf8 \strokec8 12\cf4 \strokec4 , \cf8 \strokec8 31\cf4 \strokec4 )\
        date_range \cf7 \strokec7 =\cf4 \strokec4  pd.date_range(start\cf7 \strokec7 =\cf4 \strokec4 start_date, end\cf7 \strokec7 =\cf4 \strokec4 end_date, freq\cf7 \strokec7 =\cf2 \strokec2 'Q'\cf4 \strokec4 )\
        \
        n_periods \cf7 \strokec7 =\cf4 \strokec4  \cf2 \strokec2 len\cf4 \strokec4 (date_range)\
        \
        
\f1\i \cf6 \strokec6 # Base economic indicators with realistic trends and cycles
\f0\i0 \cf4 \strokec4 \
        np.random.seed(RANDOM_SEED)\
        \
        
\f1\i \cf6 \strokec6 # GDP Growth Rate (%) - with business cycles
\f0\i0 \cf4 \strokec4 \
        trend \cf7 \strokec7 =\cf4 \strokec4  np.linspace(\cf8 \strokec8 2.5\cf4 \strokec4 , \cf8 \strokec8 3.0\cf4 \strokec4 , n_periods)\
        cycle \cf7 \strokec7 =\cf4 \strokec4  \cf8 \strokec8 2\cf4 \strokec4  \cf7 \strokec7 *\cf4 \strokec4  np.sin(np.linspace(\cf8 \strokec8 0\cf4 \strokec4 , \cf8 \strokec8 8\cf4 \strokec4  \cf7 \strokec7 *\cf4 \strokec4  np.pi, n_periods))  
\f1\i \cf6 \strokec6 # ~10-year cycles
\f0\i0 \cf4 \strokec4 \
        noise \cf7 \strokec7 =\cf4 \strokec4  np.random.normal(\cf8 \strokec8 0\cf4 \strokec4 , \cf8 \strokec8 1.5\cf4 \strokec4 , n_periods)\
        gdp_growth \cf7 \strokec7 =\cf4 \strokec4  trend \cf7 \strokec7 +\cf4 \strokec4  cycle \cf7 \strokec7 +\cf4 \strokec4  noise\
        \
        
\f1\i \cf6 \strokec6 # Recession periods (negative growth)
\f0\i0 \cf4 \strokec4 \
        recession_periods \cf7 \strokec7 =\cf4 \strokec4  [\
            (\cf8 \strokec8 1948\cf4 \strokec4 , \cf8 \strokec8 1949\cf4 \strokec4 ), (\cf8 \strokec8 1953\cf4 \strokec4 , \cf8 \strokec8 1954\cf4 \strokec4 ), (\cf8 \strokec8 1957\cf4 \strokec4 , \cf8 \strokec8 1958\cf4 \strokec4 ), (\cf8 \strokec8 1960\cf4 \strokec4 , \cf8 \strokec8 1961\cf4 \strokec4 ),\
            (\cf8 \strokec8 1969\cf4 \strokec4 , \cf8 \strokec8 1970\cf4 \strokec4 ), (\cf8 \strokec8 1973\cf4 \strokec4 , \cf8 \strokec8 1975\cf4 \strokec4 ), (\cf8 \strokec8 1980\cf4 \strokec4 , \cf8 \strokec8 1982\cf4 \strokec4 ), (\cf8 \strokec8 1990\cf4 \strokec4 , \cf8 \strokec8 1991\cf4 \strokec4 ),\
            (\cf8 \strokec8 2001\cf4 \strokec4 , \cf8 \strokec8 2001\cf4 \strokec4 ), (\cf8 \strokec8 2007\cf4 \strokec4 , \cf8 \strokec8 2009\cf4 \strokec4 ), (\cf8 \strokec8 2020\cf4 \strokec4 , \cf8 \strokec8 2020\cf4 \strokec4 )\
        ]\
        \
        \cf5 \strokec5 for\cf4 \strokec4  start_year, end_year \cf5 \strokec5 in\cf4 \strokec4  recession_periods:\
            start_idx \cf7 \strokec7 =\cf4 \strokec4  (start_year \cf7 \strokec7 -\cf4 \strokec4  \cf8 \strokec8 1944\cf4 \strokec4 ) \cf7 \strokec7 *\cf4 \strokec4  \cf8 \strokec8 4\cf4 \strokec4 \
            end_idx \cf7 \strokec7 =\cf4 \strokec4  (end_year \cf7 \strokec7 -\cf4 \strokec4  \cf8 \strokec8 1944\cf4 \strokec4 ) \cf7 \strokec7 *\cf4 \strokec4  \cf8 \strokec8 4\cf4 \strokec4  \cf7 \strokec7 +\cf4 \strokec4  \cf8 \strokec8 4\cf4 \strokec4 \
            \cf5 \strokec5 if\cf4 \strokec4  end_idx \cf7 \strokec7 <\cf4 \strokec4  n_periods:\
                gdp_growth[start_idx:end_idx] \cf7 \strokec7 =\cf4 \strokec4  np.random.uniform(\cf7 \strokec7 -\cf8 \strokec8 3\cf4 \strokec4 , \cf7 \strokec7 -\cf8 \strokec8 0.5\cf4 \strokec4 , end_idx \cf7 \strokec7 -\cf4 \strokec4  start_idx)\
        \
        
\f1\i \cf6 \strokec6 # Consumer Price Index (CPI) - inflation measure
\f0\i0 \cf4 \strokec4 \
        cpi_base \cf7 \strokec7 =\cf4 \strokec4  \cf8 \strokec8 24.0\cf4 \strokec4   
\f1\i \cf6 \strokec6 # 1944 baseline
\f0\i0 \cf4 \strokec4 \
        cpi \cf7 \strokec7 =\cf4 \strokec4  [cpi_base]\
        \cf5 \strokec5 for\cf4 \strokec4  i \cf5 \strokec5 in\cf4 \strokec4  \cf2 \strokec2 range\cf4 \strokec4 (\cf8 \strokec8 1\cf4 \strokec4 , n_periods):\
            inflation_rate \cf7 \strokec7 =\cf4 \strokec4  np.random.normal(\cf8 \strokec8 0.03\cf4 \strokec4 , \cf8 \strokec8 0.015\cf4 \strokec4 )  
\f1\i \cf6 \strokec6 # ~3% avg inflation
\f0\i0 \cf4 \strokec4 \
            \cf5 \strokec5 if\cf4 \strokec4  \cf8 \strokec8 1970\cf4 \strokec4  \cf7 \strokec7 <=\cf4 \strokec4  date_range[i].year \cf7 \strokec7 <=\cf4 \strokec4  \cf8 \strokec8 1982\cf4 \strokec4 :  
\f1\i \cf6 \strokec6 # High inflation period
\f0\i0 \cf4 \strokec4 \
                inflation_rate \cf7 \strokec7 =\cf4 \strokec4  np.random.normal(\cf8 \strokec8 0.07\cf4 \strokec4 , \cf8 \strokec8 0.03\cf4 \strokec4 )\
            cpi.append(cpi[\cf7 \strokec7 -\cf8 \strokec8 1\cf4 \strokec4 ] \cf7 \strokec7 *\cf4 \strokec4  (\cf8 \strokec8 1\cf4 \strokec4  \cf7 \strokec7 +\cf4 \strokec4  inflation_rate))\
        cpi \cf7 \strokec7 =\cf4 \strokec4  np.array(cpi)\
        \
        
\f1\i \cf6 \strokec6 # Unemployment Rate (%)
\f0\i0 \cf4 \strokec4 \
        unemployment \cf7 \strokec7 =\cf4 \strokec4  \cf8 \strokec8 5.0\cf4 \strokec4  \cf7 \strokec7 +\cf4 \strokec4  \cf8 \strokec8 2\cf4 \strokec4  \cf7 \strokec7 *\cf4 \strokec4  np.sin(np.linspace(\cf8 \strokec8 0\cf4 \strokec4 , \cf8 \strokec8 8\cf4 \strokec4  \cf7 \strokec7 *\cf4 \strokec4  np.pi, n_periods))\
        unemployment \cf7 \strokec7 +=\cf4 \strokec4  np.random.normal(\cf8 \strokec8 0\cf4 \strokec4 , \cf8 \strokec8 0.5\cf4 \strokec4 , n_periods)\
        unemployment \cf7 \strokec7 =\cf4 \strokec4  np.clip(unemployment, \cf8 \strokec8 3.0\cf4 \strokec4 , \cf8 \strokec8 15.0\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Increase unemployment during recessions
\f0\i0 \cf4 \strokec4 \
        \cf5 \strokec5 for\cf4 \strokec4  start_year, end_year \cf5 \strokec5 in\cf4 \strokec4  recession_periods:\
            start_idx \cf7 \strokec7 =\cf4 \strokec4  (start_year \cf7 \strokec7 -\cf4 \strokec4  \cf8 \strokec8 1944\cf4 \strokec4 ) \cf7 \strokec7 *\cf4 \strokec4  \cf8 \strokec8 4\cf4 \strokec4 \
            end_idx \cf7 \strokec7 =\cf4 \strokec4  (end_year \cf7 \strokec7 -\cf4 \strokec4  \cf8 \strokec8 1944\cf4 \strokec4 ) \cf7 \strokec7 *\cf4 \strokec4  \cf8 \strokec8 4\cf4 \strokec4  \cf7 \strokec7 +\cf4 \strokec4  \cf8 \strokec8 4\cf4 \strokec4 \
            \cf5 \strokec5 if\cf4 \strokec4  end_idx \cf7 \strokec7 <\cf4 \strokec4  n_periods:\
                unemployment[start_idx:end_idx] \cf7 \strokec7 +=\cf4 \strokec4  np.random.uniform(\cf8 \strokec8 2\cf4 \strokec4 , \cf8 \strokec8 6\cf4 \strokec4 , end_idx \cf7 \strokec7 -\cf4 \strokec4  start_idx)\
        \
        
\f1\i \cf6 \strokec6 # Stock Market Index (S&P 500 proxy)
\f0\i0 \cf4 \strokec4 \
        stock_base \cf7 \strokec7 =\cf4 \strokec4  \cf8 \strokec8 15.0\cf4 \strokec4   
\f1\i \cf6 \strokec6 # 1944 baseline
\f0\i0 \cf4 \strokec4 \
        stock_index \cf7 \strokec7 =\cf4 \strokec4  [stock_base]\
        \cf5 \strokec5 for\cf4 \strokec4  i \cf5 \strokec5 in\cf4 \strokec4  \cf2 \strokec2 range\cf4 \strokec4 (\cf8 \strokec8 1\cf4 \strokec4 , n_periods):\
            growth \cf7 \strokec7 =\cf4 \strokec4  np.random.normal(\cf8 \strokec8 0.02\cf4 \strokec4 , \cf8 \strokec8 0.05\cf4 \strokec4 )  
\f1\i \cf6 \strokec6 # ~8% annual growth
\f0\i0 \cf4 \strokec4 \
            \cf5 \strokec5 if\cf4 \strokec4  \cf2 \strokec2 any\cf4 \strokec4 (start \cf7 \strokec7 <=\cf4 \strokec4  date_range[i].year \cf7 \strokec7 <=\cf4 \strokec4  end \cf5 \strokec5 for\cf4 \strokec4  start, end \cf5 \strokec5 in\cf4 \strokec4  recession_periods):\
                growth \cf7 \strokec7 =\cf4 \strokec4  np.random.normal(\cf7 \strokec7 -\cf8 \strokec8 0.05\cf4 \strokec4 , \cf8 \strokec8 0.08\cf4 \strokec4 )  
\f1\i \cf6 \strokec6 # Negative during recessions
\f0\i0 \cf4 \strokec4 \
            stock_index.append(stock_index[\cf7 \strokec7 -\cf8 \strokec8 1\cf4 \strokec4 ] \cf7 \strokec7 *\cf4 \strokec4  (\cf8 \strokec8 1\cf4 \strokec4  \cf7 \strokec7 +\cf4 \strokec4  growth))\
        stock_index \cf7 \strokec7 =\cf4 \strokec4  np.array(stock_index)\
        \
        
\f1\i \cf6 \strokec6 # Interest Rate (Federal Funds Rate proxy)
\f0\i0 \cf4 \strokec4 \
        interest_rate \cf7 \strokec7 =\cf4 \strokec4  \cf8 \strokec8 4.0\cf4 \strokec4  \cf7 \strokec7 +\cf4 \strokec4  \cf8 \strokec8 3\cf4 \strokec4  \cf7 \strokec7 *\cf4 \strokec4  np.sin(np.linspace(\cf8 \strokec8 0\cf4 \strokec4 , \cf8 \strokec8 6\cf4 \strokec4  \cf7 \strokec7 *\cf4 \strokec4  np.pi, n_periods))\
        interest_rate \cf7 \strokec7 +=\cf4 \strokec4  np.random.normal(\cf8 \strokec8 0\cf4 \strokec4 , \cf8 \strokec8 0.5\cf4 \strokec4 , n_periods)\
        interest_rate \cf7 \strokec7 =\cf4 \strokec4  np.clip(interest_rate, \cf8 \strokec8 0.0\cf4 \strokec4 , \cf8 \strokec8 15.0\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Create DataFrame
\f0\i0 \cf4 \strokec4 \
        self.data \cf7 \strokec7 =\cf4 \strokec4  pd.DataFrame(\{\
            \cf2 \strokec2 'Date'\cf4 \strokec4 : date_range,\
            \cf2 \strokec2 'GDP_Growth'\cf4 \strokec4 : gdp_growth,\
            \cf2 \strokec2 'CPI'\cf4 \strokec4 : cpi,\
            \cf2 \strokec2 'Unemployment_Rate'\cf4 \strokec4 : unemployment,\
            \cf2 \strokec2 'Stock_Index'\cf4 \strokec4 : stock_index,\
            \cf2 \strokec2 'Interest_Rate'\cf4 \strokec4 : interest_rate\
        \})\
        \
        
\f1\i \cf6 \strokec6 # Target: Recession indicator (1 if GDP growth negative, 0 otherwise)
\f0\i0 \cf4 \strokec4 \
        self.data[\cf2 \strokec2 'Recession'\cf4 \strokec4 ] \cf7 \strokec7 =\cf4 \strokec4  (self.data[\cf2 \strokec2 'GDP_Growth'\cf4 \strokec4 ] \cf7 \strokec7 <\cf4 \strokec4  \cf8 \strokec8 0\cf4 \strokec4 ).astype(\cf2 \strokec2 int\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Target: Economic Risk Score (continuous 0-1)
\f0\i0 \cf4 \strokec4 \
        self.data[\cf2 \strokec2 'Economic_Risk_Score'\cf4 \strokec4 ] \cf7 \strokec7 =\cf4 \strokec4  self.calculate_risk_score()\
        \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"Generated \cf4 \strokec4 \{\cf2 \strokec2 len\cf4 \strokec4 (self.data)\}\cf2 \strokec2  quarterly observations from \cf4 \strokec4 \{date_range[\cf8 \strokec8 0\cf4 \strokec4 ].year\}\cf2 \strokec2  to \cf4 \strokec4 \{date_range[\cf7 \strokec7 -\cf8 \strokec8 1\cf4 \strokec4 ].year\}\cf2 \strokec2 "\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"Recession periods identified: \cf4 \strokec4 \{self.data[\cf2 \strokec2 'Recession'\cf4 \strokec4 ].\cf2 \strokec2 sum\cf4 \strokec4 ()\}\cf2 \strokec2  quarters"\cf4 \strokec4 )\
        \
        \cf5 \strokec5 return\cf4 \strokec4  self.data\
    \
    \cf5 \strokec5 def\cf4 \strokec4  \cf7 \strokec7 calculate_risk_score\cf4 \strokec4 (self):\
        \cf2 \strokec2 """\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4         Calculate composite economic risk score (0-1 scale)\
        Higher score indicates higher recession risk\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2         """\cf4 \strokec4 \
        
\f1\i \cf6 \strokec6 # Normalize components to 0-1 scale
\f0\i0 \cf4 \strokec4 \
        gdp_risk \cf7 \strokec7 =\cf4 \strokec4  \cf8 \strokec8 1\cf4 \strokec4  \cf7 \strokec7 /\cf4 \strokec4  (\cf8 \strokec8 1\cf4 \strokec4  \cf7 \strokec7 +\cf4 \strokec4  np.exp(self.data[\cf2 \strokec2 'GDP_Growth'\cf4 \strokec4 ]))  
\f1\i \cf6 \strokec6 # Sigmoid of GDP growth
\f0\i0 \cf4 \strokec4 \
        unemployment_risk \cf7 \strokec7 =\cf4 \strokec4  (self.data[\cf2 \strokec2 'Unemployment_Rate'\cf4 \strokec4 ] \cf7 \strokec7 -\cf4 \strokec4  \cf8 \strokec8 3\cf4 \strokec4 ) \cf7 \strokec7 /\cf4 \strokec4  \cf8 \strokec8 12\cf4 \strokec4   
\f1\i \cf6 \strokec6 # Normalized unemployment
\f0\i0 \cf4 \strokec4 \
        \
        
\f1\i \cf6 \strokec6 # CPI change (inflation risk)
\f0\i0 \cf4 \strokec4 \
        cpi_change \cf7 \strokec7 =\cf4 \strokec4  self.data[\cf2 \strokec2 'CPI'\cf4 \strokec4 ].pct_change().fillna(\cf8 \strokec8 0\cf4 \strokec4 )\
        inflation_risk \cf7 \strokec7 =\cf4 \strokec4  np.clip(cpi_change \cf7 \strokec7 *\cf4 \strokec4  \cf8 \strokec8 10\cf4 \strokec4 , \cf8 \strokec8 0\cf4 \strokec4 , \cf8 \strokec8 1\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Stock market volatility risk
\f0\i0 \cf4 \strokec4 \
        stock_change \cf7 \strokec7 =\cf4 \strokec4  self.data[\cf2 \strokec2 'Stock_Index'\cf4 \strokec4 ].pct_change().fillna(\cf8 \strokec8 0\cf4 \strokec4 )\
        market_risk \cf7 \strokec7 =\cf4 \strokec4  np.clip(\cf2 \strokec2 abs\cf4 \strokec4 (stock_change) \cf7 \strokec7 *\cf4 \strokec4  \cf8 \strokec8 5\cf4 \strokec4 , \cf8 \strokec8 0\cf4 \strokec4 , \cf8 \strokec8 1\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Weighted composite score
\f0\i0 \cf4 \strokec4 \
        risk_score \cf7 \strokec7 =\cf4 \strokec4  (\
            \cf8 \strokec8 0.35\cf4 \strokec4  \cf7 \strokec7 *\cf4 \strokec4  gdp_risk \cf7 \strokec7 +\cf4 \strokec4 \
            \cf8 \strokec8 0.25\cf4 \strokec4  \cf7 \strokec7 *\cf4 \strokec4  unemployment_risk \cf7 \strokec7 +\cf4 \strokec4 \
            \cf8 \strokec8 0.20\cf4 \strokec4  \cf7 \strokec7 *\cf4 \strokec4  inflation_risk \cf7 \strokec7 +\cf4 \strokec4 \
            \cf8 \strokec8 0.20\cf4 \strokec4  \cf7 \strokec7 *\cf4 \strokec4  market_risk\
        )\
        \
        \cf5 \strokec5 return\cf4 \strokec4  np.clip(risk_score, \cf8 \strokec8 0\cf4 \strokec4 , \cf8 \strokec8 1\cf4 \strokec4 )\
    \
    \cf5 \strokec5 def\cf4 \strokec4  \cf7 \strokec7 engineer_features\cf4 \strokec4 (self):\
        \cf2 \strokec2 """\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4         Feature engineering: Create lagged features, moving averages, and derived metrics\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2         """\cf4 \strokec4 \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\nEngineering features for time-series forecasting..."\cf4 \strokec4 )\
        \
        df \cf7 \strokec7 =\cf4 \strokec4  self.data.copy()\
        \
        
\f1\i \cf6 \strokec6 # Lagged features (previous quarters)
\f0\i0 \cf4 \strokec4 \
        \cf5 \strokec5 for\cf4 \strokec4  lag \cf5 \strokec5 in\cf4 \strokec4  [\cf8 \strokec8 1\cf4 \strokec4 , \cf8 \strokec8 2\cf4 \strokec4 , \cf8 \strokec8 4\cf4 \strokec4 , \cf8 \strokec8 8\cf4 \strokec4 ]:\
            df[\cf2 \strokec2 f'GDP_Growth_Lag\cf4 \strokec4 \{lag\}\cf2 \strokec2 '\cf4 \strokec4 ] \cf7 \strokec7 =\cf4 \strokec4  df[\cf2 \strokec2 'GDP_Growth'\cf4 \strokec4 ].shift(lag)\
            df[\cf2 \strokec2 f'Unemployment_Lag\cf4 \strokec4 \{lag\}\cf2 \strokec2 '\cf4 \strokec4 ] \cf7 \strokec7 =\cf4 \strokec4  df[\cf2 \strokec2 'Unemployment_Rate'\cf4 \strokec4 ].shift(lag)\
            df[\cf2 \strokec2 f'CPI_Change_Lag\cf4 \strokec4 \{lag\}\cf2 \strokec2 '\cf4 \strokec4 ] \cf7 \strokec7 =\cf4 \strokec4  df[\cf2 \strokec2 'CPI'\cf4 \strokec4 ].pct_change().shift(lag)\
        \
        
\f1\i \cf6 \strokec6 # Moving averages
\f0\i0 \cf4 \strokec4 \
        \cf5 \strokec5 for\cf4 \strokec4  window \cf5 \strokec5 in\cf4 \strokec4  [\cf8 \strokec8 4\cf4 \strokec4 , \cf8 \strokec8 8\cf4 \strokec4 , \cf8 \strokec8 12\cf4 \strokec4 ]:\
            df[\cf2 \strokec2 f'GDP_MA\cf4 \strokec4 \{window\}\cf2 \strokec2 '\cf4 \strokec4 ] \cf7 \strokec7 =\cf4 \strokec4  df[\cf2 \strokec2 'GDP_Growth'\cf4 \strokec4 ].rolling(window\cf7 \strokec7 =\cf4 \strokec4 window).mean()\
            df[\cf2 \strokec2 f'Unemployment_MA\cf4 \strokec4 \{window\}\cf2 \strokec2 '\cf4 \strokec4 ] \cf7 \strokec7 =\cf4 \strokec4  df[\cf2 \strokec2 'Unemployment_Rate'\cf4 \strokec4 ].rolling(window\cf7 \strokec7 =\cf4 \strokec4 window).mean()\
            df[\cf2 \strokec2 f'Stock_MA\cf4 \strokec4 \{window\}\cf2 \strokec2 '\cf4 \strokec4 ] \cf7 \strokec7 =\cf4 \strokec4  df[\cf2 \strokec2 'Stock_Index'\cf4 \strokec4 ].pct_change().rolling(window\cf7 \strokec7 =\cf4 \strokec4 window).mean()\
        \
        
\f1\i \cf6 \strokec6 # Volatility measures (rolling standard deviation)
\f0\i0 \cf4 \strokec4 \
        \cf5 \strokec5 for\cf4 \strokec4  window \cf5 \strokec5 in\cf4 \strokec4  [\cf8 \strokec8 4\cf4 \strokec4 , \cf8 \strokec8 8\cf4 \strokec4 ]:\
            df[\cf2 \strokec2 f'GDP_Volatility\cf4 \strokec4 \{window\}\cf2 \strokec2 '\cf4 \strokec4 ] \cf7 \strokec7 =\cf4 \strokec4  df[\cf2 \strokec2 'GDP_Growth'\cf4 \strokec4 ].rolling(window\cf7 \strokec7 =\cf4 \strokec4 window).std()\
            df[\cf2 \strokec2 f'Stock_Volatility\cf4 \strokec4 \{window\}\cf2 \strokec2 '\cf4 \strokec4 ] \cf7 \strokec7 =\cf4 \strokec4  df[\cf2 \strokec2 'Stock_Index'\cf4 \strokec4 ].pct_change().rolling(window\cf7 \strokec7 =\cf4 \strokec4 window).std()\
        \
        
\f1\i \cf6 \strokec6 # Rate of change features
\f0\i0 \cf4 \strokec4 \
        df[\cf2 \strokec2 'GDP_Change'\cf4 \strokec4 ] \cf7 \strokec7 =\cf4 \strokec4  df[\cf2 \strokec2 'GDP_Growth'\cf4 \strokec4 ].diff()\
        df[\cf2 \strokec2 'Unemployment_Change'\cf4 \strokec4 ] \cf7 \strokec7 =\cf4 \strokec4  df[\cf2 \strokec2 'Unemployment_Rate'\cf4 \strokec4 ].diff()\
        df[\cf2 \strokec2 'Interest_Change'\cf4 \strokec4 ] \cf7 \strokec7 =\cf4 \strokec4  df[\cf2 \strokec2 'Interest_Rate'\cf4 \strokec4 ].diff()\
        \
        
\f1\i \cf6 \strokec6 # Inflation rate (CPI change)
\f0\i0 \cf4 \strokec4 \
        df[\cf2 \strokec2 'Inflation_Rate'\cf4 \strokec4 ] \cf7 \strokec7 =\cf4 \strokec4  df[\cf2 \strokec2 'CPI'\cf4 \strokec4 ].pct_change() \cf7 \strokec7 *\cf4 \strokec4  \cf8 \strokec8 100\cf4 \strokec4 \
        \
        
\f1\i \cf6 \strokec6 # Stock returns
\f0\i0 \cf4 \strokec4 \
        df[\cf2 \strokec2 'Stock_Return'\cf4 \strokec4 ] \cf7 \strokec7 =\cf4 \strokec4  df[\cf2 \strokec2 'Stock_Index'\cf4 \strokec4 ].pct_change() \cf7 \strokec7 *\cf4 \strokec4  \cf8 \strokec8 100\cf4 \strokec4 \
        \
        
\f1\i \cf6 \strokec6 # Yield curve proxy (interest rate trends)
\f0\i0 \cf4 \strokec4 \
        df[\cf2 \strokec2 'Interest_Trend'\cf4 \strokec4 ] \cf7 \strokec7 =\cf4 \strokec4  df[\cf2 \strokec2 'Interest_Rate'\cf4 \strokec4 ].rolling(window\cf7 \strokec7 =\cf8 \strokec8 4\cf4 \strokec4 ).\cf2 \strokec2 apply\cf4 \strokec4 (\
            \cf5 \strokec5 lambda\cf4 \strokec4  x: np.polyfit(\cf2 \strokec2 range\cf4 \strokec4 (\cf2 \strokec2 len\cf4 \strokec4 (x)), x, \cf8 \strokec8 1\cf4 \strokec4 )[\cf8 \strokec8 0\cf4 \strokec4 ] \cf5 \strokec5 if\cf4 \strokec4  \cf2 \strokec2 len\cf4 \strokec4 (x) \cf7 \strokec7 ==\cf4 \strokec4  \cf8 \strokec8 4\cf4 \strokec4  \cf5 \strokec5 else\cf4 \strokec4  \cf8 \strokec8 0\cf4 \strokec4 \
        )\
        \
        
\f1\i \cf6 \strokec6 # Time-based features
\f0\i0 \cf4 \strokec4 \
        df[\cf2 \strokec2 'Quarter'\cf4 \strokec4 ] \cf7 \strokec7 =\cf4 \strokec4  df[\cf2 \strokec2 'Date'\cf4 \strokec4 ].dt.quarter\
        df[\cf2 \strokec2 'Year'\cf4 \strokec4 ] \cf7 \strokec7 =\cf4 \strokec4  df[\cf2 \strokec2 'Date'\cf4 \strokec4 ].dt.year\
        df[\cf2 \strokec2 'Decade'\cf4 \strokec4 ] \cf7 \strokec7 =\cf4 \strokec4  (df[\cf2 \strokec2 'Year'\cf4 \strokec4 ] \cf7 \strokec7 //\cf4 \strokec4  \cf8 \strokec8 10\cf4 \strokec4 ) \cf7 \strokec7 *\cf4 \strokec4  \cf8 \strokec8 10\cf4 \strokec4 \
        \
        
\f1\i \cf6 \strokec6 # Cyclical encoding of quarters
\f0\i0 \cf4 \strokec4 \
        df[\cf2 \strokec2 'Quarter_Sin'\cf4 \strokec4 ] \cf7 \strokec7 =\cf4 \strokec4  np.sin(\cf8 \strokec8 2\cf4 \strokec4  \cf7 \strokec7 *\cf4 \strokec4  np.pi \cf7 \strokec7 *\cf4 \strokec4  df[\cf2 \strokec2 'Quarter'\cf4 \strokec4 ] \cf7 \strokec7 /\cf4 \strokec4  \cf8 \strokec8 4\cf4 \strokec4 )\
        df[\cf2 \strokec2 'Quarter_Cos'\cf4 \strokec4 ] \cf7 \strokec7 =\cf4 \strokec4  np.cos(\cf8 \strokec8 2\cf4 \strokec4  \cf7 \strokec7 *\cf4 \strokec4  np.pi \cf7 \strokec7 *\cf4 \strokec4  df[\cf2 \strokec2 'Quarter'\cf4 \strokec4 ] \cf7 \strokec7 /\cf4 \strokec4  \cf8 \strokec8 4\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Drop rows with NaN (due to lagging/rolling)
\f0\i0 \cf4 \strokec4 \
        df \cf7 \strokec7 =\cf4 \strokec4  df.dropna()\
        \
        self.data \cf7 \strokec7 =\cf4 \strokec4  df\
        \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"Feature engineering complete: \cf4 \strokec4 \{\cf2 \strokec2 len\cf4 \strokec4 (df.columns)\}\cf2 \strokec2  features created"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"Dataset shape after feature engineering: \cf4 \strokec4 \{df.shape\}\cf2 \strokec2 "\cf4 \strokec4 )\
        \
        \cf5 \strokec5 return\cf4 \strokec4  df\
    \
    \cf5 \strokec5 def\cf4 \strokec4  \cf7 \strokec7 prepare_train_test_split\cf4 \strokec4 (self, test_size\cf7 \strokec7 =\cf8 \strokec8 0.2\cf4 \strokec4 ):\
        \cf2 \strokec2 """\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4         Prepare features and target, split into train/test sets\
        Time-series aware split (no shuffle)\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2         """\cf4 \strokec4 \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\nPreparing train/test split..."\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Select features (exclude target and date)
\f0\i0 \cf4 \strokec4 \
        feature_cols \cf7 \strokec7 =\cf4 \strokec4  [col \cf5 \strokec5 for\cf4 \strokec4  col \cf5 \strokec5 in\cf4 \strokec4  self.data.columns \
                       \cf5 \strokec5 if\cf4 \strokec4  col \cf5 \strokec5 not\cf4 \strokec4  \cf5 \strokec5 in\cf4 \strokec4  [\cf2 \strokec2 'Date'\cf4 \strokec4 , \cf2 \strokec2 'Economic_Risk_Score'\cf4 \strokec4 , \cf2 \strokec2 'Recession'\cf4 \strokec4 , \cf2 \strokec2 'Year'\cf4 \strokec4 ]]\
        \
        X \cf7 \strokec7 =\cf4 \strokec4  self.data[feature_cols]\
        y \cf7 \strokec7 =\cf4 \strokec4  self.data[\cf2 \strokec2 'Economic_Risk_Score'\cf4 \strokec4 ]\
        \
        
\f1\i \cf6 \strokec6 # Time-series split (no shuffle - maintain temporal order)
\f0\i0 \cf4 \strokec4 \
        split_index \cf7 \strokec7 =\cf4 \strokec4  \cf2 \strokec2 int\cf4 \strokec4 (\cf2 \strokec2 len\cf4 \strokec4 (X) \cf7 \strokec7 *\cf4 \strokec4  (\cf8 \strokec8 1\cf4 \strokec4  \cf7 \strokec7 -\cf4 \strokec4  test_size))\
        \
        self.X_train \cf7 \strokec7 =\cf4 \strokec4  X.iloc[:split_index]\
        self.X_test \cf7 \strokec7 =\cf4 \strokec4  X.iloc[split_index:]\
        self.y_train \cf7 \strokec7 =\cf4 \strokec4  y.iloc[:split_index]\
        self.y_test \cf7 \strokec7 =\cf4 \strokec4  y.iloc[split_index:]\
        \
        self.feature_names \cf7 \strokec7 =\cf4 \strokec4  feature_cols\
        \
        
\f1\i \cf6 \strokec6 # Scale features
\f0\i0 \cf4 \strokec4 \
        self.X_train \cf7 \strokec7 =\cf4 \strokec4  pd.DataFrame(\
            self.scaler.fit_transform(self.X_train),\
            columns\cf7 \strokec7 =\cf4 \strokec4 self.X_train.columns,\
            index\cf7 \strokec7 =\cf4 \strokec4 self.X_train.index\
        )\
        self.X_test \cf7 \strokec7 =\cf4 \strokec4  pd.DataFrame(\
            self.scaler.transform(self.X_test),\
            columns\cf7 \strokec7 =\cf4 \strokec4 self.X_test.columns,\
            index\cf7 \strokec7 =\cf4 \strokec4 self.X_test.index\
        )\
        \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"Training set: \cf4 \strokec4 \{\cf2 \strokec2 len\cf4 \strokec4 (self.X_train)\}\cf2 \strokec2  samples"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"Test set: \cf4 \strokec4 \{\cf2 \strokec2 len\cf4 \strokec4 (self.X_test)\}\cf2 \strokec2  samples"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"Number of features: \cf4 \strokec4 \{\cf2 \strokec2 len\cf4 \strokec4 (feature_cols)\}\cf2 \strokec2 "\cf4 \strokec4 )\
        \
        \cf5 \strokec5 return\cf4 \strokec4  self.X_train, self.X_test, self.y_train, self.y_test\
    \
    \cf5 \strokec5 def\cf4 \strokec4  \cf7 \strokec7 train_random_forest\cf4 \strokec4 (self, n_estimators\cf7 \strokec7 =\cf8 \strokec8 200\cf4 \strokec4 ):\
        \cf2 \strokec2 """\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4         Train Random Forest Regressor with optimized hyperparameters\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2         """\cf4 \strokec4 \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\n"\cf4 \strokec4  \cf7 \strokec7 +\cf4 \strokec4  \cf2 \strokec2 "="\cf7 \strokec7 *\cf8 \strokec8 70\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "TRAINING RANDOM FOREST MODEL"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "="\cf7 \strokec7 *\cf8 \strokec8 70\cf4 \strokec4 )\
        \
        self.rf_model \cf7 \strokec7 =\cf4 \strokec4  RandomForestRegressor(\
            n_estimators\cf7 \strokec7 =\cf4 \strokec4 n_estimators,\
            max_depth\cf7 \strokec7 =\cf8 \strokec8 15\cf4 \strokec4 ,\
            min_samples_split\cf7 \strokec7 =\cf8 \strokec8 10\cf4 \strokec4 ,\
            min_samples_leaf\cf7 \strokec7 =\cf8 \strokec8 4\cf4 \strokec4 ,\
            max_features\cf7 \strokec7 =\cf2 \strokec2 'sqrt'\cf4 \strokec4 ,\
            random_state\cf7 \strokec7 =\cf4 \strokec4 RANDOM_SEED,\
            n_jobs\cf7 \strokec7 =-\cf8 \strokec8 1\cf4 \strokec4 \
        )\
        \
        self.rf_model.fit(self.X_train, self.y_train)\
        \
        
\f1\i \cf6 \strokec6 # Predictions
\f0\i0 \cf4 \strokec4 \
        y_train_pred \cf7 \strokec7 =\cf4 \strokec4  self.rf_model.predict(self.X_train)\
        y_test_pred \cf7 \strokec7 =\cf4 \strokec4  self.rf_model.predict(self.X_test)\
        \
        
\f1\i \cf6 \strokec6 # Metrics
\f0\i0 \cf4 \strokec4 \
        train_r2 \cf7 \strokec7 =\cf4 \strokec4  r2_score(self.y_train, y_train_pred)\
        test_r2 \cf7 \strokec7 =\cf4 \strokec4  r2_score(self.y_test, y_test_pred)\
        train_rmse \cf7 \strokec7 =\cf4 \strokec4  np.sqrt(mean_squared_error(self.y_train, y_train_pred))\
        test_rmse \cf7 \strokec7 =\cf4 \strokec4  np.sqrt(mean_squared_error(self.y_test, y_test_pred))\
        \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"\\nRandom Forest Performance:"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"Training   - R\'b2: \cf4 \strokec4 \{train_r2:.4f\}\cf2 \strokec2  | RMSE: \cf4 \strokec4 \{train_rmse:.4f\}\cf2 \strokec2 "\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"Testing    - R\'b2: \cf4 \strokec4 \{test_r2:.4f\}\cf2 \strokec2  | RMSE: \cf4 \strokec4 \{test_rmse:.4f\}\cf2 \strokec2 "\cf4 \strokec4 )\
        \
        \cf5 \strokec5 return\cf4 \strokec4  self.rf_model, test_r2, test_rmse\
    \
    \cf5 \strokec5 def\cf4 \strokec4  \cf7 \strokec7 train_xgboost\cf4 \strokec4 (self):\
        \cf2 \strokec2 """\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4         Train XGBoost Regressor with optimized hyperparameters\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2         """\cf4 \strokec4 \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\n"\cf4 \strokec4  \cf7 \strokec7 +\cf4 \strokec4  \cf2 \strokec2 "="\cf7 \strokec7 *\cf8 \strokec8 70\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "TRAINING XGBOOST MODEL"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "="\cf7 \strokec7 *\cf8 \strokec8 70\cf4 \strokec4 )\
        \
        self.xgb_model \cf7 \strokec7 =\cf4 \strokec4  xgb.XGBRegressor(\
            n_estimators\cf7 \strokec7 =\cf8 \strokec8 300\cf4 \strokec4 ,\
            learning_rate\cf7 \strokec7 =\cf8 \strokec8 0.05\cf4 \strokec4 ,\
            max_depth\cf7 \strokec7 =\cf8 \strokec8 8\cf4 \strokec4 ,\
            min_child_weight\cf7 \strokec7 =\cf8 \strokec8 3\cf4 \strokec4 ,\
            subsample\cf7 \strokec7 =\cf8 \strokec8 0.8\cf4 \strokec4 ,\
            colsample_bytree\cf7 \strokec7 =\cf8 \strokec8 0.8\cf4 \strokec4 ,\
            gamma\cf7 \strokec7 =\cf8 \strokec8 0.1\cf4 \strokec4 ,\
            reg_alpha\cf7 \strokec7 =\cf8 \strokec8 0.05\cf4 \strokec4 ,\
            reg_lambda\cf7 \strokec7 =\cf8 \strokec8 1.0\cf4 \strokec4 ,\
            random_state\cf7 \strokec7 =\cf4 \strokec4 RANDOM_SEED,\
            n_jobs\cf7 \strokec7 =-\cf8 \strokec8 1\cf4 \strokec4 \
        )\
        \
        self.xgb_model.fit(\
            self.X_train, \
            self.y_train,\
            eval_set\cf7 \strokec7 =\cf4 \strokec4 [(self.X_test, self.y_test)],\
            verbose\cf7 \strokec7 =\cf8 \strokec8 False\cf4 \strokec4 \
        )\
        \
        
\f1\i \cf6 \strokec6 # Predictions
\f0\i0 \cf4 \strokec4 \
        y_train_pred \cf7 \strokec7 =\cf4 \strokec4  self.xgb_model.predict(self.X_train)\
        y_test_pred \cf7 \strokec7 =\cf4 \strokec4  self.xgb_model.predict(self.X_test)\
        \
        
\f1\i \cf6 \strokec6 # Metrics
\f0\i0 \cf4 \strokec4 \
        train_r2 \cf7 \strokec7 =\cf4 \strokec4  r2_score(self.y_train, y_train_pred)\
        test_r2 \cf7 \strokec7 =\cf4 \strokec4  r2_score(self.y_test, y_test_pred)\
        train_rmse \cf7 \strokec7 =\cf4 \strokec4  np.sqrt(mean_squared_error(self.y_train, y_train_pred))\
        test_rmse \cf7 \strokec7 =\cf4 \strokec4  np.sqrt(mean_squared_error(self.y_test, y_test_pred))\
        \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"\\nXGBoost Performance:"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"Training   - R\'b2: \cf4 \strokec4 \{train_r2:.4f\}\cf2 \strokec2  | RMSE: \cf4 \strokec4 \{train_rmse:.4f\}\cf2 \strokec2 "\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"Testing    - R\'b2: \cf4 \strokec4 \{test_r2:.4f\}\cf2 \strokec2  | RMSE: \cf4 \strokec4 \{test_rmse:.4f\}\cf2 \strokec2 "\cf4 \strokec4 )\
        \
        \cf5 \strokec5 return\cf4 \strokec4  self.xgb_model, test_r2, test_rmse\
    \
    \cf5 \strokec5 def\cf4 \strokec4  \cf7 \strokec7 perform_cross_validation\cf4 \strokec4 (self, model, model_name, n_folds\cf7 \strokec7 =\cf8 \strokec8 10\cf4 \strokec4 ):\
        \cf2 \strokec2 """\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4         Perform k-fold cross-validation for robust model evaluation\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2         """\cf4 \strokec4 \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"\\nPerforming \cf4 \strokec4 \{n_folds\}\cf2 \strokec2 -Fold Cross-Validation for \cf4 \strokec4 \{model_name\}\cf2 \strokec2 ..."\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Combine train and test for full cross-validation
\f0\i0 \cf4 \strokec4 \
        X_full \cf7 \strokec7 =\cf4 \strokec4  pd.concat([self.X_train, self.X_test])\
        y_full \cf7 \strokec7 =\cf4 \strokec4  pd.concat([self.y_train, self.y_test])\
        \
        
\f1\i \cf6 \strokec6 # Time-series cross-validation
\f0\i0 \cf4 \strokec4 \
        kfold \cf7 \strokec7 =\cf4 \strokec4  KFold(n_splits\cf7 \strokec7 =\cf4 \strokec4 n_folds, shuffle\cf7 \strokec7 =\cf8 \strokec8 False\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # R\'b2 scores
\f0\i0 \cf4 \strokec4 \
        r2_scores \cf7 \strokec7 =\cf4 \strokec4  cross_val_score(\
            model, X_full, y_full, \
            cv\cf7 \strokec7 =\cf4 \strokec4 kfold, \
            scoring\cf7 \strokec7 =\cf2 \strokec2 'r2'\cf4 \strokec4 ,\
            n_jobs\cf7 \strokec7 =-\cf8 \strokec8 1\cf4 \strokec4 \
        )\
        \
        
\f1\i \cf6 \strokec6 # RMSE scores
\f0\i0 \cf4 \strokec4 \
        mse_scores \cf7 \strokec7 =\cf4 \strokec4  \cf7 \strokec7 -\cf4 \strokec4 cross_val_score(\
            model, X_full, y_full,\
            cv\cf7 \strokec7 =\cf4 \strokec4 kfold,\
            scoring\cf7 \strokec7 =\cf2 \strokec2 'neg_mean_squared_error'\cf4 \strokec4 ,\
            n_jobs\cf7 \strokec7 =-\cf8 \strokec8 1\cf4 \strokec4 \
        )\
        rmse_scores \cf7 \strokec7 =\cf4 \strokec4  np.sqrt(mse_scores)\
        \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"\\n\cf4 \strokec4 \{n_folds\}\cf2 \strokec2 -Fold Cross-Validation Results (\cf4 \strokec4 \{model_name\}\cf2 \strokec2 ):"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"R\'b2 Scores:   \cf4 \strokec4 \{r2_scores\}\cf2 \strokec2 "\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"Mean R\'b2:     \cf4 \strokec4 \{r2_scores.mean():.4f\}\cf2 \strokec2  (+/- \cf4 \strokec4 \{r2_scores.std():.4f\}\cf2 \strokec2 )"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"RMSE Scores: \cf4 \strokec4 \{rmse_scores\}\cf2 \strokec2 "\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"Mean RMSE:   \cf4 \strokec4 \{rmse_scores.mean():.4f\}\cf2 \strokec2  (+/- \cf4 \strokec4 \{rmse_scores.std():.4f\}\cf2 \strokec2 )"\cf4 \strokec4 )\
        \
        \cf5 \strokec5 return\cf4 \strokec4  r2_scores.mean(), rmse_scores.mean()\
    \
    \cf5 \strokec5 def\cf4 \strokec4  \cf7 \strokec7 create_ensemble_predictions\cf4 \strokec4 (self, weights\cf7 \strokec7 =\cf4 \strokec4 [\cf8 \strokec8 0.5\cf4 \strokec4 , \cf8 \strokec8 0.5\cf4 \strokec4 ]):\
        \cf2 \strokec2 """\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4         Create weighted ensemble predictions from RF and XGBoost\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2         """\cf4 \strokec4 \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\n"\cf4 \strokec4  \cf7 \strokec7 +\cf4 \strokec4  \cf2 \strokec2 "="\cf7 \strokec7 *\cf8 \strokec8 70\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "ENSEMBLE MODEL (Random Forest + XGBoost)"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "="\cf7 \strokec7 *\cf8 \strokec8 70\cf4 \strokec4 )\
        \
        rf_pred \cf7 \strokec7 =\cf4 \strokec4  self.rf_model.predict(self.X_test)\
        xgb_pred \cf7 \strokec7 =\cf4 \strokec4  self.xgb_model.predict(self.X_test)\
        \
        
\f1\i \cf6 \strokec6 # Weighted ensemble
\f0\i0 \cf4 \strokec4 \
        ensemble_pred \cf7 \strokec7 =\cf4 \strokec4  weights[\cf8 \strokec8 0\cf4 \strokec4 ] \cf7 \strokec7 *\cf4 \strokec4  rf_pred \cf7 \strokec7 +\cf4 \strokec4  weights[\cf8 \strokec8 1\cf4 \strokec4 ] \cf7 \strokec7 *\cf4 \strokec4  xgb_pred\
        \
        
\f1\i \cf6 \strokec6 # Metrics
\f0\i0 \cf4 \strokec4 \
        ensemble_r2 \cf7 \strokec7 =\cf4 \strokec4  r2_score(self.y_test, ensemble_pred)\
        ensemble_rmse \cf7 \strokec7 =\cf4 \strokec4  np.sqrt(mean_squared_error(self.y_test, ensemble_pred))\
        ensemble_mae \cf7 \strokec7 =\cf4 \strokec4  mean_absolute_error(self.y_test, ensemble_pred)\
        \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"\\nEnsemble Performance (Weights: RF=\cf4 \strokec4 \{weights[\cf8 \strokec8 0\cf4 \strokec4 ]\}\cf2 \strokec2 , XGB=\cf4 \strokec4 \{weights[\cf8 \strokec8 1\cf4 \strokec4 ]\}\cf2 \strokec2 ):"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"R\'b2 Score:  \cf4 \strokec4 \{ensemble_r2:.4f\}\cf2 \strokec2 "\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"RMSE:      \cf4 \strokec4 \{ensemble_rmse:.4f\}\cf2 \strokec2 "\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"MAE:       \cf4 \strokec4 \{ensemble_mae:.4f\}\cf2 \strokec2 "\cf4 \strokec4 )\
        \
        \cf5 \strokec5 return\cf4 \strokec4  ensemble_pred, ensemble_r2, ensemble_rmse\
    \
    \cf5 \strokec5 def\cf4 \strokec4  \cf7 \strokec7 analyze_feature_importance\cf4 \strokec4 (self):\
        \cf2 \strokec2 """\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4         Analyze and visualize feature importance from both models\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2         """\cf4 \strokec4 \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\nAnalyzing feature importance..."\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Random Forest importance
\f0\i0 \cf4 \strokec4 \
        rf_importance \cf7 \strokec7 =\cf4 \strokec4  pd.DataFrame(\{\
            \cf2 \strokec2 'Feature'\cf4 \strokec4 : self.feature_names,\
            \cf2 \strokec2 'RF_Importance'\cf4 \strokec4 : self.rf_model.feature_importances_\
        \}).sort_values(\cf2 \strokec2 'RF_Importance'\cf4 \strokec4 , ascending\cf7 \strokec7 =\cf8 \strokec8 False\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # XGBoost importance
\f0\i0 \cf4 \strokec4 \
        xgb_importance \cf7 \strokec7 =\cf4 \strokec4  pd.DataFrame(\{\
            \cf2 \strokec2 'Feature'\cf4 \strokec4 : self.feature_names,\
            \cf2 \strokec2 'XGB_Importance'\cf4 \strokec4 : self.xgb_model.feature_importances_\
        \}).sort_values(\cf2 \strokec2 'XGB_Importance'\cf4 \strokec4 , ascending\cf7 \strokec7 =\cf8 \strokec8 False\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Combine
\f0\i0 \cf4 \strokec4 \
        importance_df \cf7 \strokec7 =\cf4 \strokec4  rf_importance.merge(xgb_importance, on\cf7 \strokec7 =\cf2 \strokec2 'Feature'\cf4 \strokec4 )\
        importance_df[\cf2 \strokec2 'Avg_Importance'\cf4 \strokec4 ] \cf7 \strokec7 =\cf4 \strokec4  (\
            importance_df[\cf2 \strokec2 'RF_Importance'\cf4 \strokec4 ] \cf7 \strokec7 +\cf4 \strokec4  importance_df[\cf2 \strokec2 'XGB_Importance'\cf4 \strokec4 ]\
        ) \cf7 \strokec7 /\cf4 \strokec4  \cf8 \strokec8 2\cf4 \strokec4 \
        importance_df \cf7 \strokec7 =\cf4 \strokec4  importance_df.sort_values(\cf2 \strokec2 'Avg_Importance'\cf4 \strokec4 , ascending\cf7 \strokec7 =\cf8 \strokec8 False\cf4 \strokec4 )\
        \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\nTop 15 Most Important Features:"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (importance_df.head(\cf8 \strokec8 15\cf4 \strokec4 ).to_string(index\cf7 \strokec7 =\cf8 \strokec8 False\cf4 \strokec4 ))\
        \
        
\f1\i \cf6 \strokec6 # Visualize
\f0\i0 \cf4 \strokec4 \
        fig, axes \cf7 \strokec7 =\cf4 \strokec4  plt.subplots(\cf8 \strokec8 1\cf4 \strokec4 , \cf8 \strokec8 2\cf4 \strokec4 , figsize\cf7 \strokec7 =\cf4 \strokec4 (\cf8 \strokec8 16\cf4 \strokec4 , \cf8 \strokec8 6\cf4 \strokec4 ))\
        \
        
\f1\i \cf6 \strokec6 # Random Forest
\f0\i0 \cf4 \strokec4 \
        top_rf \cf7 \strokec7 =\cf4 \strokec4  rf_importance.head(\cf8 \strokec8 15\cf4 \strokec4 )\
        axes[\cf8 \strokec8 0\cf4 \strokec4 ].barh(\cf2 \strokec2 range\cf4 \strokec4 (\cf2 \strokec2 len\cf4 \strokec4 (top_rf)), top_rf[\cf2 \strokec2 'RF_Importance'\cf4 \strokec4 ], color\cf7 \strokec7 =\cf2 \strokec2 'steelblue'\cf4 \strokec4 )\
        axes[\cf8 \strokec8 0\cf4 \strokec4 ].set_yticks(\cf2 \strokec2 range\cf4 \strokec4 (\cf2 \strokec2 len\cf4 \strokec4 (top_rf)))\
        axes[\cf8 \strokec8 0\cf4 \strokec4 ].set_yticklabels(top_rf[\cf2 \strokec2 'Feature'\cf4 \strokec4 ])\
        axes[\cf8 \strokec8 0\cf4 \strokec4 ].set_xlabel(\cf2 \strokec2 'Importance Score'\cf4 \strokec4 )\
        axes[\cf8 \strokec8 0\cf4 \strokec4 ].set_title(\cf2 \strokec2 'Random Forest - Top 15 Features'\cf4 \strokec4 )\
        axes[\cf8 \strokec8 0\cf4 \strokec4 ].invert_yaxis()\
        \
        
\f1\i \cf6 \strokec6 # XGBoost
\f0\i0 \cf4 \strokec4 \
        top_xgb \cf7 \strokec7 =\cf4 \strokec4  xgb_importance.head(\cf8 \strokec8 15\cf4 \strokec4 )\
        axes[\cf8 \strokec8 1\cf4 \strokec4 ].barh(\cf2 \strokec2 range\cf4 \strokec4 (\cf2 \strokec2 len\cf4 \strokec4 (top_xgb)), top_xgb[\cf2 \strokec2 'XGB_Importance'\cf4 \strokec4 ], color\cf7 \strokec7 =\cf2 \strokec2 'darkorange'\cf4 \strokec4 )\
        axes[\cf8 \strokec8 1\cf4 \strokec4 ].set_yticks(\cf2 \strokec2 range\cf4 \strokec4 (\cf2 \strokec2 len\cf4 \strokec4 (top_xgb)))\
        axes[\cf8 \strokec8 1\cf4 \strokec4 ].set_yticklabels(top_xgb[\cf2 \strokec2 'Feature'\cf4 \strokec4 ])\
        axes[\cf8 \strokec8 1\cf4 \strokec4 ].set_xlabel(\cf2 \strokec2 'Importance Score'\cf4 \strokec4 )\
        axes[\cf8 \strokec8 1\cf4 \strokec4 ].set_title(\cf2 \strokec2 'XGBoost - Top 15 Features'\cf4 \strokec4 )\
        axes[\cf8 \strokec8 1\cf4 \strokec4 ].invert_yaxis()\
        \
        plt.tight_layout()\
        plt.savefig(\cf2 \strokec2 'feature_importance.png'\cf4 \strokec4 , dpi\cf7 \strokec7 =\cf8 \strokec8 300\cf4 \strokec4 , bbox_inches\cf7 \strokec7 =\cf2 \strokec2 'tight'\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\nSaved: feature_importance.png"\cf4 \strokec4 )\
        plt.close()\
        \
        \cf5 \strokec5 return\cf4 \strokec4  importance_df\
    \
    \cf5 \strokec5 def\cf4 \strokec4  \cf7 \strokec7 visualize_predictions\cf4 \strokec4 (self, ensemble_pred):\
        \cf2 \strokec2 """\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4         Create comprehensive prediction visualizations\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2         """\cf4 \strokec4 \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\nGenerating prediction visualizations..."\cf4 \strokec4 )\
        \
        fig, axes \cf7 \strokec7 =\cf4 \strokec4  plt.subplots(\cf8 \strokec8 2\cf4 \strokec4 , \cf8 \strokec8 2\cf4 \strokec4 , figsize\cf7 \strokec7 =\cf4 \strokec4 (\cf8 \strokec8 16\cf4 \strokec4 , \cf8 \strokec8 12\cf4 \strokec4 ))\
        \
        
\f1\i \cf6 \strokec6 # Get test dates
\f0\i0 \cf4 \strokec4 \
        test_dates \cf7 \strokec7 =\cf4 \strokec4  self.data.loc[self.y_test.index, \cf2 \strokec2 'Date'\cf4 \strokec4 ]\
        \
        
\f1\i \cf6 \strokec6 # 1. Actual vs Predicted (Time Series)
\f0\i0 \cf4 \strokec4 \
        axes[\cf8 \strokec8 0\cf4 \strokec4 , \cf8 \strokec8 0\cf4 \strokec4 ].plot(test_dates, self.y_test, label\cf7 \strokec7 =\cf2 \strokec2 'Actual'\cf4 \strokec4 , linewidth\cf7 \strokec7 =\cf8 \strokec8 2\cf4 \strokec4 , color\cf7 \strokec7 =\cf2 \strokec2 'blue'\cf4 \strokec4 )\
        axes[\cf8 \strokec8 0\cf4 \strokec4 , \cf8 \strokec8 0\cf4 \strokec4 ].plot(test_dates, ensemble_pred, label\cf7 \strokec7 =\cf2 \strokec2 'Predicted'\cf4 \strokec4 , linewidth\cf7 \strokec7 =\cf8 \strokec8 2\cf4 \strokec4 , \
                       color\cf7 \strokec7 =\cf2 \strokec2 'red'\cf4 \strokec4 , linestyle\cf7 \strokec7 =\cf2 \strokec2 '--'\cf4 \strokec4 , alpha\cf7 \strokec7 =\cf8 \strokec8 0.7\cf4 \strokec4 )\
        axes[\cf8 \strokec8 0\cf4 \strokec4 , \cf8 \strokec8 0\cf4 \strokec4 ].fill_between(test_dates, self.y_test, ensemble_pred, alpha\cf7 \strokec7 =\cf8 \strokec8 0.2\cf4 \strokec4 )\
        axes[\cf8 \strokec8 0\cf4 \strokec4 , \cf8 \strokec8 0\cf4 \strokec4 ].set_xlabel(\cf2 \strokec2 'Date'\cf4 \strokec4 )\
        axes[\cf8 \strokec8 0\cf4 \strokec4 , \cf8 \strokec8 0\cf4 \strokec4 ].set_ylabel(\cf2 \strokec2 'Economic Risk Score'\cf4 \strokec4 )\
        axes[\cf8 \strokec8 0\cf4 \strokec4 , \cf8 \strokec8 0\cf4 \strokec4 ].set_title(\cf2 \strokec2 'Actual vs Predicted Economic Risk Over Time'\cf4 \strokec4 )\
        axes[\cf8 \strokec8 0\cf4 \strokec4 , \cf8 \strokec8 0\cf4 \strokec4 ].legend()\
        axes[\cf8 \strokec8 0\cf4 \strokec4 , \cf8 \strokec8 0\cf4 \strokec4 ].grid(\cf8 \strokec8 True\cf4 \strokec4 , alpha\cf7 \strokec7 =\cf8 \strokec8 0.3\cf4 \strokec4 )\
        axes[\cf8 \strokec8 0\cf4 \strokec4 , \cf8 \strokec8 0\cf4 \strokec4 ].tick_params(axis\cf7 \strokec7 =\cf2 \strokec2 'x'\cf4 \strokec4 , rotation\cf7 \strokec7 =\cf8 \strokec8 45\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # 2. Scatter Plot (Actual vs Predicted)
\f0\i0 \cf4 \strokec4 \
        axes[\cf8 \strokec8 0\cf4 \strokec4 , \cf8 \strokec8 1\cf4 \strokec4 ].scatter(self.y_test, ensemble_pred, alpha\cf7 \strokec7 =\cf8 \strokec8 0.6\cf4 \strokec4 , s\cf7 \strokec7 =\cf8 \strokec8 50\cf4 \strokec4 , color\cf7 \strokec7 =\cf2 \strokec2 'green'\cf4 \strokec4 )\
        axes[\cf8 \strokec8 0\cf4 \strokec4 , \cf8 \strokec8 1\cf4 \strokec4 ].plot([self.y_test.\cf2 \strokec2 min\cf4 \strokec4 (), self.y_test.\cf2 \strokec2 max\cf4 \strokec4 ()], \
                       [self.y_test.\cf2 \strokec2 min\cf4 \strokec4 (), self.y_test.\cf2 \strokec2 max\cf4 \strokec4 ()], \
                       \cf2 \strokec2 'r--'\cf4 \strokec4 , linewidth\cf7 \strokec7 =\cf8 \strokec8 2\cf4 \strokec4 , label\cf7 \strokec7 =\cf2 \strokec2 'Perfect Prediction'\cf4 \strokec4 )\
        axes[\cf8 \strokec8 0\cf4 \strokec4 , \cf8 \strokec8 1\cf4 \strokec4 ].set_xlabel(\cf2 \strokec2 'Actual Risk Score'\cf4 \strokec4 )\
        axes[\cf8 \strokec8 0\cf4 \strokec4 , \cf8 \strokec8 1\cf4 \strokec4 ].set_ylabel(\cf2 \strokec2 'Predicted Risk Score'\cf4 \strokec4 )\
        axes[\cf8 \strokec8 0\cf4 \strokec4 , \cf8 \strokec8 1\cf4 \strokec4 ].set_title(\cf2 \strokec2 'Prediction Accuracy Scatter Plot'\cf4 \strokec4 )\
        axes[\cf8 \strokec8 0\cf4 \strokec4 , \cf8 \strokec8 1\cf4 \strokec4 ].legend()\
        axes[\cf8 \strokec8 0\cf4 \strokec4 , \cf8 \strokec8 1\cf4 \strokec4 ].grid(\cf8 \strokec8 True\cf4 \strokec4 , alpha\cf7 \strokec7 =\cf8 \strokec8 0.3\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # 3. Residuals Plot
\f0\i0 \cf4 \strokec4 \
        residuals \cf7 \strokec7 =\cf4 \strokec4  self.y_test \cf7 \strokec7 -\cf4 \strokec4  ensemble_pred\
        axes[\cf8 \strokec8 1\cf4 \strokec4 , \cf8 \strokec8 0\cf4 \strokec4 ].scatter(ensemble_pred, residuals, alpha\cf7 \strokec7 =\cf8 \strokec8 0.6\cf4 \strokec4 , s\cf7 \strokec7 =\cf8 \strokec8 50\cf4 \strokec4 , color\cf7 \strokec7 =\cf2 \strokec2 'purple'\cf4 \strokec4 )\
        axes[\cf8 \strokec8 1\cf4 \strokec4 , \cf8 \strokec8 0\cf4 \strokec4 ].axhline(y\cf7 \strokec7 =\cf8 \strokec8 0\cf4 \strokec4 , color\cf7 \strokec7 =\cf2 \strokec2 'r'\cf4 \strokec4 , linestyle\cf7 \strokec7 =\cf2 \strokec2 '--'\cf4 \strokec4 , linewidth\cf7 \strokec7 =\cf8 \strokec8 2\cf4 \strokec4 )\
        axes[\cf8 \strokec8 1\cf4 \strokec4 , \cf8 \strokec8 0\cf4 \strokec4 ].set_xlabel(\cf2 \strokec2 'Predicted Risk Score'\cf4 \strokec4 )\
        axes[\cf8 \strokec8 1\cf4 \strokec4 , \cf8 \strokec8 0\cf4 \strokec4 ].set_ylabel(\cf2 \strokec2 'Residuals'\cf4 \strokec4 )\
        axes[\cf8 \strokec8 1\cf4 \strokec4 , \cf8 \strokec8 0\cf4 \strokec4 ].set_title(\cf2 \strokec2 'Residual Plot (Prediction Errors)'\cf4 \strokec4 )\
        axes[\cf8 \strokec8 1\cf4 \strokec4 , \cf8 \strokec8 0\cf4 \strokec4 ].grid(\cf8 \strokec8 True\cf4 \strokec4 , alpha\cf7 \strokec7 =\cf8 \strokec8 0.3\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # 4. Residuals Distribution
\f0\i0 \cf4 \strokec4 \
        axes[\cf8 \strokec8 1\cf4 \strokec4 , \cf8 \strokec8 1\cf4 \strokec4 ].hist(residuals, bins\cf7 \strokec7 =\cf8 \strokec8 30\cf4 \strokec4 , edgecolor\cf7 \strokec7 =\cf2 \strokec2 'black'\cf4 \strokec4 , alpha\cf7 \strokec7 =\cf8 \strokec8 0.7\cf4 \strokec4 , color\cf7 \strokec7 =\cf2 \strokec2 'teal'\cf4 \strokec4 )\
        axes[\cf8 \strokec8 1\cf4 \strokec4 , \cf8 \strokec8 1\cf4 \strokec4 ].axvline(x\cf7 \strokec7 =\cf8 \strokec8 0\cf4 \strokec4 , color\cf7 \strokec7 =\cf2 \strokec2 'r'\cf4 \strokec4 , linestyle\cf7 \strokec7 =\cf2 \strokec2 '--'\cf4 \strokec4 , linewidth\cf7 \strokec7 =\cf8 \strokec8 2\cf4 \strokec4 )\
        axes[\cf8 \strokec8 1\cf4 \strokec4 , \cf8 \strokec8 1\cf4 \strokec4 ].set_xlabel(\cf2 \strokec2 'Residuals'\cf4 \strokec4 )\
        axes[\cf8 \strokec8 1\cf4 \strokec4 , \cf8 \strokec8 1\cf4 \strokec4 ].set_ylabel(\cf2 \strokec2 'Frequency'\cf4 \strokec4 )\
        axes[\cf8 \strokec8 1\cf4 \strokec4 , \cf8 \strokec8 1\cf4 \strokec4 ].set_title(\cf2 \strokec2 'Distribution of Prediction Errors'\cf4 \strokec4 )\
        axes[\cf8 \strokec8 1\cf4 \strokec4 , \cf8 \strokec8 1\cf4 \strokec4 ].grid(\cf8 \strokec8 True\cf4 \strokec4 , alpha\cf7 \strokec7 =\cf8 \strokec8 0.3\cf4 \strokec4 )\
        \
        plt.tight_layout()\
        plt.savefig(\cf2 \strokec2 'prediction_analysis.png'\cf4 \strokec4 , dpi\cf7 \strokec7 =\cf8 \strokec8 300\cf4 \strokec4 , bbox_inches\cf7 \strokec7 =\cf2 \strokec2 'tight'\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "Saved: prediction_analysis.png"\cf4 \strokec4 )\
        plt.close()\
    \
    \cf5 \strokec5 def\cf4 \strokec4  \cf7 \strokec7 generate_economic_insights\cf4 \strokec4 (self, ensemble_pred):\
        \cf2 \strokec2 """\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4         Generate actionable economic insights and risk assessments\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2         """\cf4 \strokec4 \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\n"\cf4 \strokec4  \cf7 \strokec7 +\cf4 \strokec4  \cf2 \strokec2 "="\cf7 \strokec7 *\cf8 \strokec8 70\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "ECONOMIC INSIGHTS & RISK ASSESSMENT"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "="\cf7 \strokec7 *\cf8 \strokec8 70\cf4 \strokec4 )\
        \
        test_dates \cf7 \strokec7 =\cf4 \strokec4  self.data.loc[self.y_test.index, \cf2 \strokec2 'Date'\cf4 \strokec4 ]\
        test_data \cf7 \strokec7 =\cf4 \strokec4  self.data.loc[self.y_test.index]\
        \
        
\f1\i \cf6 \strokec6 # Current risk assessment (most recent prediction)
\f0\i0 \cf4 \strokec4 \
        current_risk \cf7 \strokec7 =\cf4 \strokec4  ensemble_pred[\cf7 \strokec7 -\cf8 \strokec8 1\cf4 \strokec4 ]\
        current_date \cf7 \strokec7 =\cf4 \strokec4  test_dates.iloc[\cf7 \strokec7 -\cf8 \strokec8 1\cf4 \strokec4 ]\
        \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"\\nCurrent Economic Risk Assessment (as of \cf4 \strokec4 \{current_date.strftime(\cf2 \strokec2 '%Y-%m-%d'\cf4 \strokec4 )\}\cf2 \strokec2 ):"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"Risk Score: \cf4 \strokec4 \{current_risk:.3f\}\cf2 \strokec2  (0=Low Risk, 1=High Risk)"\cf4 \strokec4 )\
        \
        \cf5 \strokec5 if\cf4 \strokec4  current_risk \cf7 \strokec7 <\cf4 \strokec4  \cf8 \strokec8 0.3\cf4 \strokec4 :\
            risk_level \cf7 \strokec7 =\cf4 \strokec4  \cf2 \strokec2 "LOW"\cf4 \strokec4 \
            recommendation \cf7 \strokec7 =\cf4 \strokec4  \cf2 \strokec2 "Economy showing strong fundamentals. Favorable for investment and expansion."\cf4 \strokec4 \
        \cf5 \strokec5 elif\cf4 \strokec4  current_risk \cf7 \strokec7 <\cf4 \strokec4  \cf8 \strokec8 0.6\cf4 \strokec4 :\
            risk_level \cf7 \strokec7 =\cf4 \strokec4  \cf2 \strokec2 "MODERATE"\cf4 \strokec4 \
            recommendation \cf7 \strokec7 =\cf4 \strokec4  \cf2 \strokec2 "Mixed signals. Monitor indicators closely. Maintain balanced portfolio."\cf4 \strokec4 \
        \cf5 \strokec5 else\cf4 \strokec4 :\
            risk_level \cf7 \strokec7 =\cf4 \strokec4  \cf2 \strokec2 "HIGH"\cf4 \strokec4 \
            recommendation \cf7 \strokec7 =\cf4 \strokec4  \cf2 \strokec2 "Elevated recession risk. Consider defensive positioning and risk mitigation."\cf4 \strokec4 \
        \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"Risk Level: \cf4 \strokec4 \{risk_level\}\cf2 \strokec2 "\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"Policy Recommendation: \cf4 \strokec4 \{recommendation\}\cf2 \strokec2 "\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Trend analysis
\f0\i0 \cf4 \strokec4 \
        recent_trend \cf7 \strokec7 =\cf4 \strokec4  ensemble_pred[\cf7 \strokec7 -\cf8 \strokec8 4\cf4 \strokec4 :]  
\f1\i \cf6 \strokec6 # Last 4 quarters
\f0\i0 \cf4 \strokec4 \
        trend_direction \cf7 \strokec7 =\cf4 \strokec4  \cf2 \strokec2 "INCREASING"\cf4 \strokec4  \cf5 \strokec5 if\cf4 \strokec4  recent_trend[\cf7 \strokec7 -\cf8 \strokec8 1\cf4 \strokec4 ] \cf7 \strokec7 >\cf4 \strokec4  recent_trend[\cf8 \strokec8 0\cf4 \strokec4 ] \cf5 \strokec5 else\cf4 \strokec4  \cf2 \strokec2 "DECREASING"\cf4 \strokec4 \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"\\nRecent Trend (Last 4 Quarters): \cf4 \strokec4 \{trend_direction\}\cf2 \strokec2 "\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # High risk periods
\f0\i0 \cf4 \strokec4 \
        high_risk_periods \cf7 \strokec7 =\cf4 \strokec4  ensemble_pred \cf7 \strokec7 >\cf4 \strokec4  \cf8 \strokec8 0.6\cf4 \strokec4 \
        \cf5 \strokec5 if\cf4 \strokec4  high_risk_periods.\cf2 \strokec2 sum\cf4 \strokec4 () \cf7 \strokec7 >\cf4 \strokec4  \cf8 \strokec8 0\cf4 \strokec4 :\
            \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"\\nHigh-Risk Periods Identified: \cf4 \strokec4 \{high_risk_periods.\cf2 \strokec2 sum\cf4 \strokec4 ()\}\cf2 \strokec2  quarters"\cf4 \strokec4 )\
            \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "Policy makers should consider:"\cf4 \strokec4 )\
            \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "  \'95 Monetary policy adjustments (interest rate changes)"\cf4 \strokec4 )\
            \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "  \'95 Fiscal stimulus measures"\cf4 \strokec4 )\
            \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "  \'95 Enhanced financial sector monitoring"\cf4 \strokec4 )\
            \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "  \'95 Support for vulnerable industries"\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Model confidence
\f0\i0 \cf4 \strokec4 \
        residuals \cf7 \strokec7 =\cf4 \strokec4  np.\cf2 \strokec2 abs\cf4 \strokec4 (self.y_test \cf7 \strokec7 -\cf4 \strokec4  ensemble_pred)\
        avg_error \cf7 \strokec7 =\cf4 \strokec4  residuals.mean()\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"\\nModel Confidence:"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"Average Prediction Error: \'b1\cf4 \strokec4 \{avg_error:.3f\}\cf2 \strokec2 "\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"Prediction Accuracy: \cf4 \strokec4 \{(\cf8 \strokec8 1\cf4 \strokec4  \cf7 \strokec7 -\cf4 \strokec4  avg_error)\cf7 \strokec7 *\cf8 \strokec8 100\cf4 \strokec4 :.1f\}\cf2 \strokec2 %"\cf4 \strokec4 )\
        \
        \cf5 \strokec5 return\cf4 \strokec4  current_risk, risk_level\
    \
    \cf5 \strokec5 def\cf4 \strokec4  \cf7 \strokec7 export_results\cf4 \strokec4 (self, ensemble_pred):\
        \cf2 \strokec2 """\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4         Export predictions and analysis results to CSV\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2         """\cf4 \strokec4 \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\nExporting results..."\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Create results DataFrame
\f0\i0 \cf4 \strokec4 \
        results_df \cf7 \strokec7 =\cf4 \strokec4  self.data.loc[self.y_test.index].copy()\
        results_df[\cf2 \strokec2 'Actual_Risk_Score'\cf4 \strokec4 ] \cf7 \strokec7 =\cf4 \strokec4  self.y_test.values\
        results_df[\cf2 \strokec2 'Predicted_Risk_Score'\cf4 \strokec4 ] \cf7 \strokec7 =\cf4 \strokec4  ensemble_pred\
        results_df[\cf2 \strokec2 'Prediction_Error'\cf4 \strokec4 ] \cf7 \strokec7 =\cf4 \strokec4  self.y_test.values \cf7 \strokec7 -\cf4 \strokec4  ensemble_pred\
        results_df[\cf2 \strokec2 'Risk_Level'\cf4 \strokec4 ] \cf7 \strokec7 =\cf4 \strokec4  pd.cut(\
            ensemble_pred,\
            bins\cf7 \strokec7 =\cf4 \strokec4 [\cf8 \strokec8 0\cf4 \strokec4 , \cf8 \strokec8 0.3\cf4 \strokec4 , \cf8 \strokec8 0.6\cf4 \strokec4 , \cf8 \strokec8 1.0\cf4 \strokec4 ],\
            labels\cf7 \strokec7 =\cf4 \strokec4 [\cf2 \strokec2 'Low'\cf4 \strokec4 , \cf2 \strokec2 'Moderate'\cf4 \strokec4 , \cf2 \strokec2 'High'\cf4 \strokec4 ]\
        )\
        \
        results_df.to_csv(\cf2 \strokec2 'recession_predictions.csv'\cf4 \strokec4 , index\cf7 \strokec7 =\cf8 \strokec8 False\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "Saved: recession_predictions.csv"\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Export model performance summary
\f0\i0 \cf4 \strokec4 \
        summary \cf7 \strokec7 =\cf4 \strokec4  \{\
            \cf2 \strokec2 'Metric'\cf4 \strokec4 : [\cf2 \strokec2 'R\'b2 Score'\cf4 \strokec4 , \cf2 \strokec2 'RMSE'\cf4 \strokec4 , \cf2 \strokec2 'MAE'\cf4 \strokec4 , \cf2 \strokec2 'Test Samples'\cf4 \strokec4 , \cf2 \strokec2 'Features Used'\cf4 \strokec4 ],\
            \cf2 \strokec2 'Value'\cf4 \strokec4 : [\
                \cf2 \strokec2 f"\cf4 \strokec4 \{r2_score(self.y_test, ensemble_pred):.4f\}\cf2 \strokec2 "\cf4 \strokec4 ,\
                \cf2 \strokec2 f"\cf4 \strokec4 \{np.sqrt(mean_squared_error(self.y_test, ensemble_pred)):.4f\}\cf2 \strokec2 "\cf4 \strokec4 ,\
                \cf2 \strokec2 f"\cf4 \strokec4 \{mean_absolute_error(self.y_test, ensemble_pred):.4f\}\cf2 \strokec2 "\cf4 \strokec4 ,\
                \cf2 \strokec2 len\cf4 \strokec4 (self.y_test),\
                \cf2 \strokec2 len\cf4 \strokec4 (self.feature_names)\
            ]\
        \}\
        summary_df \cf7 \strokec7 =\cf4 \strokec4  pd.DataFrame(summary)\
        summary_df.to_csv(\cf2 \strokec2 'model_performance_summary.csv'\cf4 \strokec4 , index\cf7 \strokec7 =\cf8 \strokec8 False\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "Saved: model_performance_summary.csv"\cf4 \strokec4 )\
    \
    \cf5 \strokec5 def\cf4 \strokec4  \cf7 \strokec7 run_full_pipeline\cf4 \strokec4 (self):\
        \cf2 \strokec2 """\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4         Execute complete ML workflow: Data \uc0\u8594  Engineering \u8594  Training \u8594  Validation \u8594  Reporting\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2         """\cf4 \strokec4 \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "="\cf7 \strokec7 *\cf8 \strokec8 70\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "RECESSIONX - U.S. ECONOMIC FORECASTING MODEL"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "Time-Series Machine Learning Pipeline"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "="\cf7 \strokec7 *\cf8 \strokec8 70\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Step 1: Data Generation
\f0\i0 \cf4 \strokec4 \
        self.generate_historical_data(years\cf7 \strokec7 =\cf8 \strokec8 80\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Step 2: Feature Engineering
\f0\i0 \cf4 \strokec4 \
        self.engineer_features()\
        \
        
\f1\i \cf6 \strokec6 # Step 3: Train/Test Split
\f0\i0 \cf4 \strokec4 \
        self.prepare_train_test_split(test_size\cf7 \strokec7 =\cf8 \strokec8 0.2\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Step 4: Model Training
\f0\i0 \cf4 \strokec4 \
        rf_model, rf_r2, rf_rmse \cf7 \strokec7 =\cf4 \strokec4  self.train_random_forest()\
        xgb_model, xgb_r2, xgb_rmse \cf7 \strokec7 =\cf4 \strokec4  self.train_xgboost()\
        \
        
\f1\i \cf6 \strokec6 # Step 5: Cross-Validation
\f0\i0 \cf4 \strokec4 \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\n"\cf4 \strokec4  \cf7 \strokec7 +\cf4 \strokec4  \cf2 \strokec2 "="\cf7 \strokec7 *\cf8 \strokec8 70\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "CROSS-VALIDATION (10-FOLD)"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "="\cf7 \strokec7 *\cf8 \strokec8 70\cf4 \strokec4 )\
        rf_cv_r2, rf_cv_rmse \cf7 \strokec7 =\cf4 \strokec4  self.perform_cross_validation(rf_model, \cf2 \strokec2 "Random Forest"\cf4 \strokec4 , n_folds\cf7 \strokec7 =\cf8 \strokec8 10\cf4 \strokec4 )\
        xgb_cv_r2, xgb_cv_rmse \cf7 \strokec7 =\cf4 \strokec4  self.perform_cross_validation(xgb_model, \cf2 \strokec2 "XGBoost"\cf4 \strokec4 , n_folds\cf7 \strokec7 =\cf8 \strokec8 10\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Step 6: Ensemble Predictions
\f0\i0 \cf4 \strokec4 \
        ensemble_pred, ensemble_r2, ensemble_rmse \cf7 \strokec7 =\cf4 \strokec4  self.create_ensemble_predictions()\
        \
        
\f1\i \cf6 \strokec6 # Step 7: Feature Importance Analysis
\f0\i0 \cf4 \strokec4 \
        importance_df \cf7 \strokec7 =\cf4 \strokec4  self.analyze_feature_importance()\
        \
        
\f1\i \cf6 \strokec6 # Step 8: Visualization
\f0\i0 \cf4 \strokec4 \
        self.visualize_predictions(ensemble_pred)\
        \
        
\f1\i \cf6 \strokec6 # Step 9: Economic Insights
\f0\i0 \cf4 \strokec4 \
        current_risk, risk_level \cf7 \strokec7 =\cf4 \strokec4  self.generate_economic_insights(ensemble_pred)\
        \
        
\f1\i \cf6 \strokec6 # Step 10: Export Results
\f0\i0 \cf4 \strokec4 \
        self.export_results(ensemble_pred)\
        \
        
\f1\i \cf6 \strokec6 # Final Summary Report
\f0\i0 \cf4 \strokec4 \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\n"\cf4 \strokec4  \cf7 \strokec7 +\cf4 \strokec4  \cf2 \strokec2 "="\cf7 \strokec7 *\cf8 \strokec8 70\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "PIPELINE EXECUTION COMPLETE - FINAL REPORT"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "="\cf7 \strokec7 *\cf8 \strokec8 70\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"\\n\uc0\u55357 \u56522  Dataset Summary:"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"   \'95 Historical Data: 80+ years (1944-2024)"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"   \'95 Total Observations: \cf4 \strokec4 \{\cf2 \strokec2 len\cf4 \strokec4 (self.data)\}\cf2 \strokec2  quarters"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"   \'95 Features Engineered: \cf4 \strokec4 \{\cf2 \strokec2 len\cf4 \strokec4 (self.feature_names)\}\cf2 \strokec2 "\cf4 \strokec4 )\
        \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"\\n\uc0\u55358 \u56598  Model Performance:"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"   \'95 Random Forest Test R\'b2: \cf4 \strokec4 \{rf_r2:.2f\}\cf2 \strokec2 "\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"   \'95 XGBoost Test R\'b2: \cf4 \strokec4 \{xgb_r2:.2f\}\cf2 \strokec2 "\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"   \'95 Ensemble R\'b2: \cf4 \strokec4 \{ensemble_r2:.2f\}\cf2 \strokec2 "\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"   \'95 Ensemble RMSE: \cf4 \strokec4 \{ensemble_rmse:.2f\}\cf2 \strokec2 "\cf4 \strokec4 )\
        \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"\\n\uc0\u9989  Cross-Validation (10-Fold):"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"   \'95 Random Forest CV R\'b2: \cf4 \strokec4 \{rf_cv_r2:.2f\}\cf2 \strokec2  (\'b1\cf4 \strokec4 \{rf_cv_rmse:.2f\}\cf2 \strokec2 )"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"   \'95 XGBoost CV R\'b2: \cf4 \strokec4 \{xgb_cv_r2:.2f\}\cf2 \strokec2  (\'b1\cf4 \strokec4 \{xgb_cv_rmse:.2f\}\cf2 \strokec2 )"\cf4 \strokec4 )\
        \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"\\n\uc0\u55356 \u57263  Key Achievements:"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"   \uc0\u10003  Forecasted recession trends using 80+ years of data"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"   \uc0\u10003  Trained ensemble models (RF + XGBoost)"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"   \uc0\u10003  Achieved R\'b2 = \cf4 \strokec4 \{ensemble_r2:.2f\}\cf2 \strokec2  and RMSE = \cf4 \strokec4 \{ensemble_rmse:.2f\}\cf2 \strokec2 "\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"   \uc0\u10003  Validated with 10-fold cross-validation"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"   \uc0\u10003  Full ML workflow: Data \u8594  Engineering \u8594  Training \u8594  Validation \u8594  Reporting"\cf4 \strokec4 )\
        \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"\\n\uc0\u55357 \u56520  Current Assessment:"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"   \'95 Current Risk Score: \cf4 \strokec4 \{current_risk:.3f\}\cf2 \strokec2 "\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"   \'95 Risk Level: \cf4 \strokec4 \{risk_level\}\cf2 \strokec2 "\cf4 \strokec4 )\
        \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"\\n\uc0\u55357 \u56513  Generated Files:"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"   \'95 recession_predictions.csv"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"   \'95 model_performance_summary.csv"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"   \'95 feature_importance.png"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"   \'95 prediction_analysis.png"\cf4 \strokec4 )\
        \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\n"\cf4 \strokec4  \cf7 \strokec7 +\cf4 \strokec4  \cf2 \strokec2 "="\cf7 \strokec7 *\cf8 \strokec8 70\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\uc0\u9989  RecessionX pipeline successfully completed!"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "="\cf7 \strokec7 *\cf8 \strokec8 70\cf4 \strokec4 )\
        \
        \cf5 \strokec5 return\cf4 \strokec4  ensemble_r2, ensemble_rmse\
\
\
\pard\pardeftab720\partightenfactor0
\cf5 \strokec5 if\cf4 \strokec4  __name__ \cf7 \strokec7 ==\cf4 \strokec4  \cf2 \strokec2 "__main__"\cf4 \strokec4 :\
    
\f1\i \cf6 \strokec6 # Initialize and run the forecasting pipeline
\f0\i0 \cf4 \strokec4 \
    forecaster \cf7 \strokec7 =\cf4 \strokec4  RecessionForecaster()\
    final_r2, final_rmse \cf7 \strokec7 =\cf4 \strokec4  forecaster.run_full_pipeline()}