# _*_coding:utf8_*_
"""
Requires:
(1)对数据的要求：因变量字段为'y'；干预变量字段为'group'，时间变量字段为'period'，交叉变量字段为'g_and_p'，其他变量自定义即可
"""
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from statsmodels.genmod.generalized_linear_model import GLM
import statsmodels.api as sm
import pandas as pd
import numpy as np
# 导入连接工具

# 导入as_pandas工具，可以将获取到的数据转化为 dataframe或者 Series

import warnings

warnings.filterwarnings('ignore')


# 异常类定义
class DidError(Exception):
    pass


# 异常类定义
class MatchError(Exception):
    pass


# Match 类
class Match(object):
    def __init__(self, control_data, test_data, sample_method):
        """
        # 初始化类成员：变量
        :param control_data: 对照组数据
        :param test_data: 实验组数据
        :param sample_method: 抽样方式，0 表示「不放回抽样」， 1 表示「有放回抽样」
        """
        self.__control_data = control_data
        self.__test_data = test_data
        self.__sample_method = sample_method
        self.check()

    def check(self):
        """
        # 不放回抽样的检查：要求 对照组数据量 至少是 实验组数据量的 5 倍「需要进一步讨论」
        :return:
        """
        if self.__sample_method == 0:
            control_size = self.__control_data.shape[0]  # .shape[0] gives the number of rows
            test_size = self.__test_data.shape[0]  # (gives dimensions of the array)
            rate = control_size / test_size
            if rate < 1:  # Question: Why rate < 1 instead of < 5?
                raise MatchError('Non-replacement sampling requires that the size of data in the control group be at '
                                 'least 5 times the size of data in the experimental group, please check the data you '
                                 'input.')
            else:
                return True

    def eu_distance_matching(self):
        """
        # match 过程：利用欧式距离 one_to_one 给出匹配数据loc
        :return: 列表，匹配数据loc
        """
        ss = StandardScaler()
        control_index = self.__control_data.index
        print(control_index)
        test_index = self.__test_data.index
        control_data = pd.DataFrame(ss.fit_transform(self.__control_data))
        test_data = pd.DataFrame(ss.fit_transform(self.__test_data))
        control_data.index = control_index
        test_data.index = test_index
        # control_data = self.__control_data
        # test_data = self.__test_data
        match_list = []
        for item in test_data.iterrows():
            inx = item[1]
            data_rows_num = control_data.shape[0]
            diffmat = np.tile(inx, (data_rows_num, 1)) - control_data
            sqdiffmat = np.power(diffmat, 2)
            sqdistances = sqdiffmat.sum(axis=1)
            distances = np.power(sqdistances, 0.5)
            # print(distances)
            one = distances.idxmin()  # find the minimum distance to be the best match
            # Return index of first occurrence of minimum over requested axis.
            match_list.append(one)
            if self.__sample_method == 0:
                control_data = control_data.drop(one, axis=0)  # without replacement
            else:
                pass
        return match_list

    def cos_distance_matching(self):
        """
        # 过程：利用余弦距离 one_to_one 给出匹配数据loc
        :return: 列表，匹配数据loc
        """
        match_list = []
        control_data = self.__control_data
        test_data = self.__test_data
        for item in test_data.iterrows():
            inx = item[1]
            data_rows_num = control_data.shape[0]
            inx_data = np.tile(inx, (data_rows_num, 1))
            num_1 = inx_data * control_data
            num = num_1.sum(axis=1)
            denom_1_1 = np.power(inx_data, 2)
            denom_1 = np.power(denom_1_1.sum(axis=1), 0.5)
            denom_2_1 = np.power(control_data, 2)
            denom_2 = np.power(denom_2_1.sum(axis=1), 0.5)
            cos = num / (denom_1 * denom_2)
            sim = 0.5 + 0.5 * cos  # cosine range [-1,1], normalize to [0,1]
            # print(sim)
            one = sim.idxmax()
            match_list.append(one)
            if self.__sample_method == 0:
                control_data = control_data.drop(one, axis=0)
            else:
                pass
        return match_list

    def propensity_score_matching(self, treated_sample, control_sample):
        """
        # 过程：利用倾向得分 one_to_one 给出匹配数据loc
        :param treated_sample: 实验组数据（包含'group 字段'）
        :param control_sample: 对照组数据（包含'group 字段'）
        :return: 列表，匹配数据loc
        """
        # 两个筛选的数据集为回归基础数据
        data_p = pd.concat([treated_sample, control_sample])
        # step1：确定回归方程
        y_field = ['group']
        print(y_field)
        x_field = list(data_p.columns)  # get he name of the index
        print(x_field)
        x_field.remove(y_field[0])
        formula = '{} ~ {}'.format(y_field[0], '+'.join(x_field))  # y = x1 + x2 + x3 + ...
        print('Formula:\n' + formula)
        # step2：拟合方程并给出 score
        y_samp = data_p[y_field]
        x_samp = data_p[x_field]
        glm = GLM(y_samp, x_samp, family=sm.families.Binomial(sm.families.links.logit))  # 逻辑回归模型
        res = glm.fit()
        data_p['score'] = res.predict(x_samp)
        # step3：针对每一个实验组得分，给出一个匹配的对照组得分
        test_scores = data_p[data_p[y_field[0]] == 1][['score']]
        print(test_scores)
        ctrl_scores = data_p[data_p[y_field[0]] == 0][['score']]
        match_list = []
        for item in test_scores.iterrows():
            # print('here:', ctrl_scores.index)
            inx = item[1]  # test score
            # print(ctrl_scores.shape)
            data_rows_num = ctrl_scores.shape[0]
            inx_data = np.tile(inx, (data_rows_num, 1))
            # print(inx_data.shape)
            diff_score = abs(inx_data - ctrl_scores)
            # print(diff_score)
            one = list(diff_score.idxmin())[0]  # index of min
            match_list.append(one)
            if self.__sample_method == 0:
                ctrl_scores = ctrl_scores.drop(one, axis=0)
            else:
                pass
        return match_list


# DID 类
class Did(object):

    def __init__(self, df):
        """
        :param df:
        """
        self.df = df

    def data_process(self):
        """
        # 处理数据，分别得到 实验组数据 以及对照组数据
        :return: 实验组数据： test_data； 对照组数据：control_data
        """
        data = df.fillna(0)
        # data[data < 0] = 0
        control_data = data[data['group'] == 0]
        test_data = data[data['group'] == 1]
        return control_data, test_data

    def match(self, drop_columns, distance_method, sample_method):
        """
        # 选取和实验组用户具有相同趋势（同质用户）的用户群体
        :param drop_columns: 除特征之外的其他字段
        :param sample_method: 抽样方式，0 表示「不放回抽样」， 1 表示「有放回抽样」
        :param distance_method: 计算用户间距离的方法，默认为'cos'，表示余弦距离；'eu'表示欧式距离；'psm'表示倾向性匹配得分Propensity score matching；
        :return: 和实验组用户具有相同趋势（同质用户）的用户群体, 全部用户群体
        """
        control_data, test_data = self.data_process()
        # print(control_data)
        """
        drop_columns: data without 'group'
        drops: data with 'group'
        """
        dropControl_wo_group = control_data.drop(drop_columns, axis=1)
        dropTest_wo_group = test_data.drop(drop_columns, axis=1)
        mat = Match(dropControl_wo_group, dropTest_wo_group, sample_method)  # initiate the class
        if distance_method == 'eu':
            match_loc = mat.eu_distance_matching()
        elif distance_method == 'cos':
            match_loc = mat.cos_distance_matching()
        elif distance_method == 'psm':
            # drops = ['y', 'group', 'period', 'g_and_p']
            drops = ['y', 'period', 'g_and_p', 'Unnamed: 0', 'sex', 'consultations', 'age']
            treated_sample = test_data.drop(drops, axis=1)
            control_sample = control_data.drop(drops, axis=1)
            treated_sample['year'] - 2005
            control_sample['year'] - 2005
            match_loc = mat.propensity_score_matching(treated_sample, control_sample)
        else:
            raise DidError('The correct param value should be cos or eu, please check the param you input.)')
        print('match_loc:', match_loc)
        match_list = [control_data.loc[row] for row in match_loc]  # create new control group
        result = pd.DataFrame(match_list)
        match_data = result.drop_duplicates()
        print('The repeated num is', (result.shape[0] - match_data.shape[0]))
        print(match_data)
        regression_data = pd.concat([test_data, match_data])
        print(regression_data)
        return match_data, regression_data

    def linear_regression(self, regression_data, x_columns):
        """
        # 线性回归求系数
        :param regression_data: 回归数据
        :param x_columns: 自变量（包括干预变量、时间变量、交叉变量、控制变量等一切想回归的变量）
        :return: 回归结果
        """
        y_data = regression_data['y']
        x_data = regression_data[x_columns]
        # sklearn 线性回归
        # model = linear_model.LinearRegression()
        # model.fit(x_data, y_data)
        # print(model.coef_)
        # print(model.intercept_)

        # statsmodel 包线性回归
        x_data = sm.add_constant(x_data)  # adding a constant
        model1 = sm.OLS(y_data, x_data).fit()
        print(model1.summary())


if __name__ == '__main__':
    # 对数据的要求：因变量(dependent variable)字段为'y'；干预变量字段为'group'，时间变量字段为'period'，交叉变量字段为'g_and_p'，其他变量自定义即可
    df = pd.read_csv("femaleVisitsToPhysician.csv")
    # 0 before treatment, 1 after treatment
    df['period'] = 0
    df['period'] = df['period'].where(df.year < 2010, 1)
    # 0 control group, 1 treatment group
    df['group'] = 0
    df['group'] = df['group'].where(df.age > 16, 1)
    # g_and_p
    df['g_and_p'] = 0
    df['g_and_p'] = df['group'] * df['period']
    df['y'] = df['perCapita']

    # 定义 match 阶段不需要的字段
    drop_column = ['group', 'y', 'period', 'g_and_p', 'Unnamed: 0', 'sex', 'consultations', 'age']
    # 定义回归阶段需要的自变量
    x_column = ['group', 'period', 'g_and_p']
    did = Did(df)
    reg_data = did.match(drop_column, 'cos', 1)[1]  # return match_data, regression_data, then get regression_data
    did.linear_regression(reg_data, x_column)
