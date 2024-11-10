import pandas as pd
import openpyxl
import requests
import json

# work with 1C over REST Odata
def getOdataRest1C(entity, filter=None, select=None):
    url = f'http://localhost/api/odata/standard.odata/Catalog_{entity}'
    headers = dict(Accept='application/json')
    params = dict()
    if filter is not None: params['$filter'] = filter 
    if select is not None: params['$select'] = select
    try:
        response = requests.get(url, headers=headers, params=params) # run request
    except:
        response = []
    try: jsonresp = response.json()
    except: raise Exception(response.text) 
    if 'odata.error' in jsonresp: raise Exception(jsonresp['odata.error']['message']['value'])
    return jsonresp['value'] # return array

def setOdataRest1C(entity, data):
    url = f'http://localhost/api/odata/standard.odata/Catalog_{entity}?$format=json'
    headers = dict(Accept='application/json')
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        jsonresp = response.json()
    except:
        #raise Exception(response.text)
        jsonresp=[]
    #if 'odata.error' in jsonresp: raise Exception(jsonresp['odata.error']['message']['value'])
    return jsonresp
# ///////////////////////////////////////////////////////////////////////////////


def search_in_df(df, search_string, search_num=None):
    # regex_pattern = '.'.join(f'[^{char}]?' for char in search_string)
    # filtered_df = df[df['ДетальАртикул'].str.contains(regex_pattern, na=False, regex=True)]
    # filtered_df = df[(df['ДетальАртикул'].str.contains(search_string)) & (df['ПорядковыйНомер'] == search_num)]
    filtered_df = df[df['ДетальАртикул'].str.contains(search_string)]
    if (search_num is not None) & (filtered_df.empty == False):
        filtered_df = filtered_df[filtered_df['ПорядковыйНомер'] == search_num]
        # filtered_df1 = filtered_df[filtered_df['ПорядковыйНомер'] == search_num]
        # if filtered_df1.empty == False:
        #     filtered_df = filtered_df1

    return filtered_df


# file_path_xls = '../ДеталиПоПлануДляРазрешенныхЗаказов.xlsx'
# text_find = 'АМ116.15.00.901'
# num_find = 0
#
# # load dataframe from excel
# df = pd.read_excel(file_path_xls)
# # or load dataframe from 1C
# # json_data = getOdataRest1C('ДеталиПоПлануДляРазрешенныхЗаказов')
# # df = pd.json_normalize(json_data)
#
# # examples search in dataframe
# search_result = search_in_df(df, text_find, num_find)
# print(text_find + ' with number ' + str(num_find) + ':')
# if search_result.empty == False:
#     print(search_result)
# else:
#     print(' not found')
