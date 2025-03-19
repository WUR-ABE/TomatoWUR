# script to calculate metrics
# quite simple insert predicted list or array and gt list or array and the metrics will be calculated
# it is indeed a bit cumbersome that the error is calculated many times, but this makes it easer to understand
# the formula

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

class Metrics():

    def __init__(self,y_pred,gt) -> None:
        print('N = %d'%len(y_pred))
        self.y_pred = np.array(y_pred)
        self.gt = np.array(gt)

        self.MEAN()
        self.MAE()
        self.MSE()
        self.RMSE()
        self.MAPE()
        # self.regression()
        pass

    def return_dataframe(self):
        """Return a dataframe with the metrics"""
        data = {'mean':[self.mean],'mean_std':[self.mean_std],'MAE':[self.mae],'MAE_std':[self.mae_std],'MSE':[self.mse],'MSE_std':[self.mse_std],'RMSE':[self.rmse],'RMSE_std':[self.rmse_std],'MAPE':[self.mape],'MAPE_std':[self.mape_std]}
        return pd.DataFrame(data)

    def MEAN(self):
        self.mean = (self.y_pred-self.gt).mean()
        self.mean_std = (self.y_pred-self.gt).std()
        print('Mean Error %0.2f std %.2f'%(self.mean, self.mean_std))

    def MAE(self):
        # https://en.wikipedia.org/wiki/Mean_absolute_error
        self.mae = np.abs(self.y_pred-self.gt).mean()
        self.mae_std = np.abs(self.y_pred-self.gt).std()
        print('Mean Absolute Error %0.2f std %.2f'%(self.mae, self.mae_std))

    def MSE(self):
        # https://en.wikipedia.org/wiki/Mean_squared_error
        self.mse = np.mean(np.power(self.y_pred-self.gt,2))
        self.mse_std = np.std(np.power(self.y_pred-self.gt,2))
        print('Mean Squared Error %0.2f std %.2f'%(self.mse, self.mse_std))

    def RMSE(self):
        # https://en.wikipedia.org/wiki/Root-mean-square_deviation
        self.rmse = np.sqrt(np.mean(np.power(self.y_pred-self.gt,2)))
        # self.mse_std = np.std((self.y_pred-self.gt)^2)
        self.rmse_std = np.nan
        print('Root Mean Squared Error %0.2f std %.2f'%(self.rmse, self.rmse_std))

    def MAPE(self):
        # https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
        self.mape = np.mean(np.abs((self.y_pred-self.gt)/self.gt)*100)
        self.mape_std = np.std(np.abs((self.y_pred-self.gt)/self.gt)*100)
        print('Mean Absolute Percentage Error %0.2f[percentage] std %.2f'%(self.mape, self.mape_std))

    ##R2 = 1-SSE / SST
    def r2(self, y_pred,gt):
        """Custom r2 to check values of .score()"""
        sse = np.sum(np.power(y_pred-gt,2))
        sst = np.sum(np.power(gt-gt.mean(),2))
        score = 1-sse/sst
        return score

    def regression(self):
        x=self.y_pred.reshape((-1,1))
        y=self.gt
        fit_intercept = False

        from sklearn.linear_model import LinearRegression
        model = LinearRegression(fit_intercept=fit_intercept)#, normalize=False)
        model.fit(x,y)

        if not fit_intercept:
            equation =  '%.3f*x'%(model.coef_)
        elif model.intercept_<0:
            equation =  '%.3f*x-%0.2f'%(model.coef_,model.intercept_)
        else:
            equation =  '%.3f*x+%0.2f'%(model.coef_,model.intercept_)
        print(equation)
        r_sq = model.score(x, y)
        print(f"coefficient of determination: {r_sq}")


        print(self.r2(model.predict(x),self.gt))
        
        from sklearn.metrics import r2_score
        x=x.squeeze()
        # plt.scatter(x,y)
        # plt.grid()
        # fpr equal axis
        # plt.xlim(0,np.max((x,y))+50)
        # plt.ylim(0,np.max((x,y))+50)
        
        # plt.xlabel("Predicted FW [gram]")
        # plt.ylabel("Ground truth [gram]")

        # Create sequence of 100 numbers from 0 to 100 
        # xseq = np.linspace(min(x), max(x), num=100)

        # Plot regression line
        # plt.plot(xseq, model.coef_* xseq+model.intercept_, "--",color="k", lw=2.5)
        # plt.title('$R^2$=%0.2f'%r_sq)
        # plt.show()

        # self.y_pred = model.predict(np.expand_dims(self.y_pred, axis=1))
        # self.MEAN()
        # self.MAE()
        # self.MSE()
        # self.RMSE()
        # self.MAPE()
        return equation, round(r_sq,2)

    def regression_with_filter(self,df,groupid='',x_axis_name = "y_pred",y_axis_name= "GT", xlabel=None, ylabel=None):
        """Either do regression using a filter, or to create a plot for each group id"""
        for unique_id in sorted(df[groupid].unique()):
            print(unique_id)
            gt_array= df[df[groupid]==unique_id][y_axis_name].dropna().values
            y_pred= df[df[groupid]==unique_id][x_axis_name].dropna().values

            ## potential do regression
            # self.y_pred = y_pred
            # self.gt = gt_array
            # self.regression()

            plt.scatter(y_pred,gt_array)
        plt.grid()
        # f0r equal axis
        # plt.xlim(0,750)
        # plt.ylim(0,750)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.legend(sorted(list(df[groupid].unique())))

        self.regression()

        plt.show()

    def regression_with_names(self,df,parts=[],legend_names=[], xlabel=None, ylabel=None):
        """So imagine if the data is not seperated by a groupid, but by different column names, then this functin could be usefull
        input
        -----
        df = pandas data frame with all data
        parts = list of lists with [[x_axis_name_1, y_axis_name_1],[x_axis_name_2, y_axis_name_2], ...]
            for example [[pred_max_height, gt_height], [pred_avg_height, gt_height]]
        legend_names = list with names ['one','two']
        xlabel = str of x-axis [pred height  [cm]
        xyabel = str of y-axis [gt height  [cm]]


        """
        colors=['tab:blue','tab:orange','tab:green']
        factor = 480

        max_obs = 0
        for i,x in enumerate(parts):
            x_axis_name =x[0]
            y_axis_name =x[1]

            df_temp = df[[x_axis_name,y_axis_name]].dropna()
            x_axis_data= df_temp[x_axis_name].values
            y_axis_data= df_temp[y_axis_name].values

            ## potential do regression
            self.y_pred = x_axis_data
            self.gt = y_axis_data
            plt.scatter(x_axis_data,y_axis_data, label=x_axis_name)
            eq, r2 = self.regression()
            plt.text(0,factor-i*0.15*factor,str('y='+eq+'\n$R^2$='+str(r2)),color=colors[i])
            print("error for %s"%x)
            self.MAE()
            self.MAPE()
            if max(x_axis_data.max(),y_axis_data.max())>max_obs:
                max_obs = max(x_axis_data.max(),y_axis_data.max())

        plt.grid()
        # for equal axis
        # plt.xlim(0,750)
        # plt.ylim(0,750)
        # Create sequence of 100 numbers from 0 to 100 
        xseq = np.linspace(0,int(max_obs*1.1), num=100)
        # Plot regression line
        plt.plot(xseq, xseq, "--",color="k", lw=2.5)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.legend(loc="lower right")

        # self.regression()
        plt.show()

        

        


# if __name__=='__main__':
    # y_pred= [1,1,2,2,3,3,4,4,5,5,6,6]
    # gt = [1,2,2,3,3,4,4,5,5,6,6,7]
    # ## example 1
    # y_pred = np.genfromtxt('x2.txt')
    # gt=np.genfromtxt('y.txt')
    # obj = Metrics(y_pred,gt)
    # obj.regression()

    ## example 2
    # folder = Path('W:\\PSG\\GlastuinbouwProjecten\\414_GreenhouseTechnology\\Marrewijk, Bart\\AGROS')
    # file_name = folder / '2023-xx-xx_agros_komkommer_metingen.xlsx'
    # df = pd.read_excel(str(file_name),sheet_name='python_pandas_pred').dropna()
    # names=['pred_weight','gt_weight']
    
    # # def f(df):
    # #     import math
    # #     return 0.835*df["length_cm"].values*np.power(df["width_cm"].values*math.pi,2)/(4*math.pi)
    # # # names=['pred_w','gt_width']
    # # # names=['pred_fw','gt_fw']
    # # y_pred = f(df)

    # gt = df[names[1]].values
    # y_pred = df[names[0]].values


    # obj = Metrics(y_pred,gt)
    # obj.regression()
    # df["y_pred"]=y_pred
    # obj.regression_with_filter(df,'compartment',y_axis_name=names[1])
    
    ##example 3
    # folder = Path('dummy_data') / 'loss_combined.csv'
    # df = pd.read_csv(str(folder)).dropna()
    # obj = Metrics([],[])
    # obj.regression_with_filter(df,'iteration','loss','miou')

    ## example 4; using column names
    # folder = Path(r'W:\PROJECTS\VisionRoboticsData\AGROS_Autonomous_Greenhouse\DepthCams\ValidationTrial\18\output')
    # file_name = folder / 'cucumber_traits.xlsx'
    # df = pd.read_excel(str(file_name),sheet_name='length_sametime_pandas')#.dropna()
    # part1 = ['days_303','width_increase_303']
    # part2 = ['days_308','width_increase_308']
    # legend_names = ['303','reg','308','reg']
    # obj = Metrics([],[])
    # obj.regression_with_names(df,parts=[part1,part2],legend_names=legend_names)