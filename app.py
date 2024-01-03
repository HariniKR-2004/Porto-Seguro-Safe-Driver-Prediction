from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import matplotlib
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
import pandas as pd
import seaborn as sns
import bnlearn as bn
import pickle as pk
matplotlib.use("Agg")
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/plots')
def plots():
    # Load your data and process it
    data = pd.read_csv("train 2.csv")
    data = data.drop(['ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04',
                      'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09',
                     'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14',
                      'ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin',
                      'ps_calc_19_bin', 'ps_calc_20_bin'], axis=1)
    data_head = data.head(10)
    # Generate the pie chart
    claim_count = len(data[data['target'] == 1])
    no_claim_count = len(data[data['target'] == 0])

    labels = ['Claim', 'No Claim']
    sizes = [claim_count, no_claim_count]
    colors = ['lightcoral', 'lightskyblue']

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Target Variable Distribution')

    # Save the pie chart as an image and convert it to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    pie_chart = base64.b64encode(buffer.read()).decode()

    # Create a correlation heatmap
    corr_matrix = data.corr()
    plt.figure(figsize=(8, 8))
    sns.heatmap(corr_matrix, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')

    # Save the heatmap as an image and convert it to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    heatmap_image = base64.b64encode(buffer.read()).decode()

    # Create box plots for numerical features
    numerical_features = ['ps_ind_01',
                          'ps_ind_02_cat', 'ps_ind_03', 'ps_ind_15']
    num_features = len(numerical_features)
    num_cols = 2
    num_rows = (num_features + 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    axes = axes.flatten()

    for i, feature in enumerate(numerical_features):
        ax = axes[i]
        sns.boxplot(x='target', y=feature, data=data, palette="Set3", ax=ax)
        ax.set_title(f'{feature} Distribution by Target')
        ax.set_xlabel('Target')
        ax.set_ylabel(feature)

    for i in range(num_features, num_cols * num_rows):
        fig.delaxes(axes[i])

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    box_plots = base64.b64encode(buffer.read()).decode()

    # Create scatter plots for numerical features
    numerical_features = ['ps_ind_01',
                          'ps_ind_02_cat', 'ps_ind_03', 'ps_ind_15']
    pair_plot = sns.pairplot(data=data, vars=numerical_features,
                             hue='target', palette="Set2", plot_kws={'s': 5})

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    scatter_plots = base64.b64encode(buffer.read()).decode()

    # Create a pivot table and stacked bar chart
    pivot_table = pd.pivot_table(data, values='id', index=[
                                 'ps_car_01_cat', 'ps_car_02_cat'], columns='target', aggfunc='count', fill_value=0)
    stacked_bar_chart = pivot_table.plot(
        kind='bar', stacked=True, colormap="Set3")
    plt.title('Stacked Bar Chart for Categorical Features')
    plt.ylabel('Count')

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    stacked_bar_chart = base64.b64encode(buffer.read()).decode()

    return render_template('plots.html', data_head=data_head, pie_chart=pie_chart, heatmap_image=heatmap_image, box_plots=box_plots, scatter_plots=scatter_plots, stacked_bar_chart=stacked_bar_chart)


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        data = request.get_json()
        entered_list = data.get('entered_list', [])
        colName = ['id', 'ps_ind_01', 'ps_ind_02_cat', 'ps_ind_03', 'ps_ind_04_cat',
                   'ps_ind_05_cat', 'ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin',
                   'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin',
                   'ps_ind_13_bin', 'ps_ind_14', 'ps_ind_15', 'ps_ind_16_bin',
                   'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_reg_01', 'ps_reg_02', 'ps_reg_03',
                   'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat',
                   'ps_car_05_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat',
                   'ps_car_09_cat', 'ps_car_10_cat', 'ps_car_11_cat', 'ps_car_11',
                   'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15']
        # Load your Bayesian network model
        with open('model.pkl', 'rb') as f:
            lmodel = pk.load(f)

        # Create a dictionary of evidence
        evidence = {colName[i]: entered_list[i] for i in range(
            len(colName)) if colName[i] not in ['ps_ind_14', 'id']}

        # Perform Bayesian network inference and get the result
        result = bn.inference.fit(
            lmodel, variables=['target'], evidence=evidence)

        # Determine the predicted target (1 or 0)
        y_pred = 1 if max(result.values) == result.values[1] else 0
        formatted_result = f"The predicted output is : {y_pred}"

        return jsonify({'result': formatted_result})

    return render_template('prediction.html')


@app.route('/pate', methods=['GET', 'POST'])
def pate():
    if request.method == 'POST':
        data = request.get_json()
        entered_list = data.get('entered_list', [])
        colName = ['id', 'ps_ind_01', 'ps_ind_02_cat', 'ps_ind_03', 'ps_ind_04_cat',
                   'ps_ind_05_cat', 'ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin',
                   'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin',
                   'ps_ind_13_bin', 'ps_ind_14', 'ps_ind_15', 'ps_ind_16_bin',
                   'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_reg_01', 'ps_reg_02', 'ps_reg_03',
                   'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat',
                   'ps_car_05_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat',
                   'ps_car_09_cat', 'ps_car_10_cat', 'ps_car_11_cat', 'ps_car_11',
                   'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15']
        with open('m1.pkl', 'rb') as f:
            lmodel1 = pk.load(f)
        with open('m2.pkl', 'rb') as f:
            lmodel2 = pk.load(f)
        with open('m3.pkl', 'rb') as f:
            lmodel3 = pk.load(f)
        with open('m4.pkl', 'rb') as f:
            lmodel4 = pk.load(f)
        with open('m5.pkl', 'rb') as f:
            lmodel5 = pk.load(f)
        k = []
        s = bn.inference.fit(lmodel1, variables=['target'], evidence={
                             colName[i]: entered_list[i] for i in range(colName) if colName[i] not in ['id', 'ps_car_10_cat']})
        k.append(s.values)
        s = bn.inference.fit(lmodel2, variables=['target'], evidence={
                             colName[i]: entered_list[i] for i in range(colName) if colName[i] not in ['id', 'ps_car_10_cat']})
        k.append(s.values)
        s = bn.inference.fit(lmodel3, variables=['target'], evidence={
                             colName[i]: entered_list[i] for i in range(colName) if colName[i] not in ['ps_ind_10_bin', 'id']})
        k.append(s.values)
        s = bn.inference.fit(lmodel4, variables=['target'], evidence={
                             colName[i]: entered_list[i] for i in range(colName) if colName[i] not in ['ps_ind_10_bin', 'id']})
        k.append(s.values)
        s = bn.inference.fit(lmodel5, variables=['target'], evidence={
                             colName[i]: entered_list[i] for i in range(colName) if colName[i] not in ['ps_car_10_cat', 'id']})
        k.append(s.values)
        ytemp = []
        for i in k:
            if(max(i) == i[0]):
                ytemp.append(0)
            else:
                ytemp.append(1)
        ypred = max(ytemp)
        formatted_result = f"The predicted output is : {ypred}"

        return jsonify({'result': formatted_result})
    return render_template('pate.html')


if __name__ == '__main__':
    app.run(debug=True)
