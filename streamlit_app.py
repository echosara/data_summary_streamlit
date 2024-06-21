import streamlit as st
import pandas as pd
import numpy as np
import math
from pathlib import Path
from itertools import product
import plotly.express as px
import plotly.graph_objects as go

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='ATE data summary',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_gdp_data():
    """Grab GDP data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """

    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME = Path(__file__).parent/'data/gdp_data.csv'
    raw_gdp_df = pd.read_csv(DATA_FILENAME)

    MIN_YEAR = 1960
    MAX_YEAR = 2022

    # The data above has columns like:
    # - Country Name
    # - Country Code
    # - [Stuff I don't care about]
    # - GDP for 1960
    # - GDP for 1961
    # - GDP for 1962
    # - ...
    # - GDP for 2022
    #
    # ...but I want this instead:
    # - Country Name
    # - Country Code
    # - Year
    # - GDP
    #
    # So let's pivot all those year-columns into two: Year and GDP
    gdp_df = raw_gdp_df.melt(
        ['Country Code'],
        [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        'Year',
        'GDP',
    )

    # Convert years from string to integers
    gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])

    return gdp_df
fig = go.Figure()
fig.update_layout(
    width=1200,
    height=800,
    bargap = 0.2,
    xaxis_title = 'Module num',
    yaxis_title = 'Efficiency/\%',
    font = dict(
        size = 28
    )
)
color_list = ['#0000ff','#00ff00','#ff0000','#ff00ff','#000000']
data_update_list = st.sidebar.file_uploader("Upload your result report", type=["csv", "xlsx"],accept_multiple_files=True)
file_num = len(data_update_list)
df_1_col = ['Module','Threshold','Efficiency','Shutdown']
df_3_col = ['VCC UVLO ON LVL/V', 'VCC UVLO ON P/F',' VCC UVLO OFF LVL/V', 'VCC UVLO OFF P/F','VCC HYS/V','VCC HYS P/F','EN INPUT HIGH LVL/V', 'EN INPUT HIGH P/F', 'EN INPUT LOW LVL/V', 'EN INPUT LOW P/F']
column_name3 = ['VOUT/V','IOUT/A','VIN/V','IIN/A','TMON/dreg','IMON/A','IVCC/mA','EFFiciency/\%','Efficiency P/F']
column_name2 = ['1p2_','0p8_','0p6_','0p4_']
column_name1 = ['A0B1_','A1B0_','A1B1_']
eff_column_raw = list(product(column_name1,column_name2,column_name3))
eff_column_list = []
for eff_column in eff_column_raw:
    eff_column_list.append(''.join(list(eff_column)))
df_5_col = ['IVCC shutdown/uA','IVCC shutdown P/F']
df_6_col = ['IVCC HIZ/mA','IVCC HIZ P/F']
result_summary_columns = df_1_col+df_3_col+eff_column_list+df_5_col+df_6_col
ylab_list = st.sidebar.multiselect('统计变量',result_summary_columns)
result_summary = pd.DataFrame(columns=result_summary_columns)
for data_update in data_update_list:
    if data_update is not None:
        df = pd.read_excel(data_update)
        rows_with_content = df.notnull().any(axis=1).sum()
        module_count =round( (rows_with_content-5)/16 ) # 根据 Excel长度计算模块数量
        df_1 = df.iloc[0:module_count,12:15] # summary
        df_2 = df.iloc[0:module_count,16:17] # shutdown summary 
        df_3 = df.iloc[module_count+1:module_count*2+1,13:23] # threshold
        df_3 = df_3.to_numpy()
        df_3 = pd.DataFrame(df_3)
        # 将dataframe的列名设置为所需的格式
        df_3.columns = ['VCC UVLO ON LVL/V', 'VCC UVLO ON P/F',' VCC UVLO OFF LVL/V', 'VCC UVLO OFF P/F','VCC HYS/V','VCC HYS P/F','EN INPUT HIGH LVL/V', 'EN INPUT HIGH P/F', 'EN INPUT LOW LVL/V', 'EN INPUT LOW P/F']
        df_4 = df.iloc[2*module_count+2:module_count*14+2,16:25] # efficiency 
        df_5 = df.iloc[14*module_count+4:module_count*15+4,13:15] # shutdown
        df_5 = df_5.to_numpy()
        df_5 = pd.DataFrame(df_5)
        df_5.columns = ['IVCC shutdown/uA','IVCC shutdown P/F']
        df_6 = df.iloc[15*module_count+7:module_count*16+7,13:15] # HIZ
        df_6 = df_6.to_numpy()
        df_6 = pd.DataFrame(df_6)
        df_6.columns = ['IVCC HIZ/mA','IVCC HIZ P/F']


        array_eff = df_4.to_numpy()
        reshape_array_eff = array_eff.reshape(module_count*3,36)
        # print(reshape_array_eff)
        # 将reshape_array_eff数组拆分为四个部分，分别表示A0B1，A1B0，A1B1
        array_eff_A0B1 = reshape_array_eff[0:module_count,:]
        array_eff_A1B0 = reshape_array_eff[module_count:module_count*2,:]
        array_eff_A1B1 = reshape_array_eff[module_count*2:module_count*3,:]
        array_eff_hsk = np.hstack((array_eff_A0B1,array_eff_A1B0,array_eff_A1B1))
        df_eff = pd.DataFrame(array_eff_hsk)
        df_eff.columns = eff_column_list
        df_post = pd.concat([df_1,df_2,df_3,df_eff,df_5,df_6],axis=1)
        for i in range(module_count):
            df_post.iloc[i,0] = data_update.name+'-'+str(i+1)
        result_summary = pd.concat([result_summary,df_post],axis=0,ignore_index=True)
    else:
        st.write('请上传数据')
# st.dataframe(result_summary)
if len(data_update_list)>0:
    if len(ylab_list)>0:
        i=0
        for ylab in ylab_list:
            fig = px.histogram(result_summary, x=ylab,color_discrete_sequence = ['blue'],opacity=0.75, marginal='box',nbins=2*round((result_summary[ylab].max()-result_summary[ylab].min())),range_x=[60,90])
            # fig = px.histogram(result_summary, x=ylab,color_discrete_sequence = ['red'],opacity=0.75, marginal='box',nbins=2*round((result_summary[ylab].max()-result_summary[ylab].min())),range_x=[60,90])
        i+=1
        # st.plotly_chart(fig)
        max_limit = st.number_input('efficiency max limit',85)
        min_limit = st.number_input('efficiency min limit',65)
        limit_between = result_summary[ylab].between(min_limit,max_limit)
        fig.add_vrect(
            x0=max_limit, x1=min_limit,
            fillcolor="LightSalmon", opacity=0.5,
            editable=True,
            layer="below", line_width=0,
        ),
        limit_percent = limit_between.sum()/len(result_summary)*100
        st.write('在',min_limit,'到',max_limit,'之间的数据占比为',limit_percent,'\%')
