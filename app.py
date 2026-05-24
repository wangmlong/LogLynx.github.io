import numpy as np
from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import io
import matplotlib
import base64
from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestRegressor
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sys
# 兼容补丁：修复 MT19937 反序列化问题
from numpy.random import MT19937

class _FakeMT19937Module:
    MT19937 = MT19937

# 伪造旧模块路径，让 pickle 能找到
sys.modules['numpy.random._mt19937'] = _FakeMT19937Module()

import warnings

warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

shap.initjs()

matplotlib.use('Agg')

app = Flask(__name__, template_folder='../html/templates')

# 加载模型
model_name = "MLP"
model = joblib.load(f'static/best_model_{model_name}.joblib')

# 加载训练数据
data_train = pd.read_csv("static/不分组训练-外部验证计算.csv", encoding='gb18030')
data_train = data_train.dropna()
# 数据预处理
# 分离特征和目标变量
x = data_train.drop('Total_Survival3', axis=1)
y = data_train["Total_Survival3"]

select_features = ['Chemical_injury', 'MGD', 'Viral_ocular_diseases_history',
                       'Systemic_autoimmune_history', 'Presence_of_corneal_neovascularization']
x = x[select_features]

@app.route('/')
def index():
    return render_template('index.html', columns=select_features)


@app.route('/predict', methods=['POST'])
def predict():
    # 获取用户输入的数据
    data = request.json
    input_data = pd.DataFrame([data], columns=select_features)
    # 预测
    # prediction = model.predict(input_data.to_numpy())
    prediction_proba = round(model.predict_proba(input_data.to_numpy())[0, 1], 4)
    prediction = np.array([(prediction_proba >= 0.37).astype(int)])

    # SHAP 分析
    # SHAP Explainer

    if model_name in ["SVM", "KNN", "NB", "LR", "AdaBoost", "XGBoost"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_data)
        print(shap_values.shape)
        if isinstance(shap_values, shap.Explanation):
            shap_values = shap_values[0]  # 选择第一个样本的 SHAP 值
        else:
            # 如果 shap_values 是一个 NumPy 数组，转换为 Explanation 对象
            shap_values = shap.Explanation(values=shap_values.values[0],
                                           base_values=shap_values.base_values[0],
                                           data=input_data.values[0],
                                           feature_names=select_features)

        # 绘制 Waterfall 图
        plt.ioff()  # 关闭交互模式
        fig, ax = plt.subplots(figsize=(12, 5))
        shap.plots.waterfall(shap_values, show=False)
        plt.savefig("waterfall_plot.png", bbox_inches='tight', dpi=400)
        # 将图像转换为 base64 格式
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        # 关闭图形
        plt.close(fig)

        # 绘制 Force Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.force_plot(explainer.expected_value, shap_values.values, input_data, text_rotation=15, show=False,
                        matplotlib=True)
        plt.savefig("force_plot.png", bbox_inches='tight', dpi=400)
        # 将图像转换为 base64 格式
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        force_plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        # 关闭图形
        plt.close(fig)
    elif model_name in ["SVM", "KNN", "MLP", "NB", "LR", "AdaBoost"]:
        explainer = shap.KernelExplainer(model.predict, x[:5])
        shap_values = explainer(input_data)
        print(shap_values.shape)
        if isinstance(shap_values, shap.Explanation):
            shap_values = shap_values[0]  # 选择第一个样本的 SHAP 值
        else:
            # 如果 shap_values 是一个 NumPy 数组，转换为 Explanation 对象
            shap_values = shap.Explanation(values=shap_values.values[0],
                                           base_values=shap_values.base_values[0],
                                           data=input_data.values[0],
                                           feature_names=select_features)

        # 绘制 Waterfall 图
        plt.ioff()  # 关闭交互模式
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_values, show=False)
        plt.savefig("waterfall_plot.png", bbox_inches='tight', dpi=400)
        # 将图像转换为 base64 格式
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        # 关闭图形
        plt.close(fig)

        # 绘制 Force Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.force_plot(explainer.expected_value, shap_values.values, input_data, text_rotation=15, show=False,
                        matplotlib=True)
        plt.savefig("force_plot.png", bbox_inches='tight', dpi=400)
        # 将图像转换为 base64 格式
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        force_plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        # 关闭图形
        plt.close(fig)

    return jsonify(
        {'prediction': prediction.tolist(), 'prediction_proba': round(prediction_proba * 100, 2), 'plot': plot_url,
         'force_plot': force_plot_url})

if __name__ == '__main__':
    app.run(debug=True)
