from pandas import DataFrame
from matplotlib import pyplot
import pandas as pd
import numpy as np

from xgboost import XGBClassifier, XGBRegressor, DMatrix


class $OperatorClass$(Operator):

    def run(self, inputs: dict, conf: dict, context: dict) -> dict:
        self.logger.info("xgb v2 operator start run....")
        title_images = []
        title_tables = []
        train_output = {}

        train_data = inputs.get("train_data", None)
        assert train_data is not None, "train_data field can not be blank"
        assert isinstance(train_data, pd.DataFrame), "train_data type is not DataFrame"
        assert const.LABEL_COLUMN in train_data.columns, f"train_data must contains {LABEL_COLUMN} column"
        train_feature_columns = utils.get_valid_features(train_data.columns)
        self.set_meta_features(train_feature_columns)

        eval_data = inputs.get("eval_data", None)
        assert eval_data is not None, "eval_data field can not be blank"
        assert isinstance(eval_data, pd.DataFrame), "eval_data type is not DataFrame"
        assert const.LABEL_COLUMN in eval_data.columns, f"eval_data must contains {LABEL_COLUMN} column"
        eval_feature_columns = utils.get_valid_features(eval_data.columns)

        utils.assert_both_same_columns(train_feature_columns, eval_feature_columns)

        self.logger.info("start train xgboost------------------------------")

        self.logger.info(f"train data size: {len(train_data)}")
        self.logger.info(
            f"train data distribution: {{0:{len(train_data[train_data[const.LABEL_COLUMN] == 0])}, 1:{len(train_data[train_data[const.LABEL_COLUMN] == 1])}}}")
        self.logger.info(f"eval data size: {len(eval_data)}")
        self.logger.info(
            f"eval data distribution: {{0:{len(eval_data[eval_data[const.LABEL_COLUMN] == 0])}, 1:{len(eval_data[eval_data[const.LABEL_COLUMN] == 1])}}}")

        if "missing" in conf["init_params"]:
            conf["init_params"].pop("missing")

        xgb_model = XGBRegressor(**conf["init_params"]) if "reg" in conf["init_params"].get("objective",
                                                                                            "") else XGBClassifier(
            **conf["init_params"])
        # xgb_model.fit(train_data[train_feature_columns], train_data[LABEL_COLUMN], eval_set=[(eval_data[eval_feature_columns], eval_data[LABEL_COLUMN])],
        #               **conf["fit_params"])

        if conf.get("enable_grid_search", False):
            self.logger.info("xgb grid_search fit...")
            xgb_model = utils.gridsearch(xgb_model, train_data, eval_data, train_feature_columns, conf,
                                         self.base_info.graph_name, title_tables, title_images)
        else:
            self.logger.info("xgb base fit...")
            xgb_model.fit(train_data[train_feature_columns], train_data[const.LABEL_COLUMN],
                          eval_set=[(eval_data[train_feature_columns], eval_data[const.LABEL_COLUMN])],
                          **conf["fit_params"])

        self.logger.info(f"best score: {xgb_model.best_score}")
        self.logger.info(
            f"model score: {xgb_model.score(eval_data[train_feature_columns], eval_data[const.LABEL_COLUMN])}")

        ## 评估

        self.logger.info('-*--------------------train eval--------------------------------*')
        train_prop = xgb_model.get_booster().predict(DMatrix(train_data[train_feature_columns]))
        train_metric_output = utils.metric_run(test_y=train_data[const.LABEL_COLUMN], y_prop=train_prop)
        title_tables.append(("train_data_metric", train_metric_output["ks_bucket"]))

        self.logger.info('-*--------------------test eval---------------------------------*')
        eval_prop = xgb_model.get_booster().predict(DMatrix(eval_data[train_feature_columns]))
        eval_metric_output = utils.metric_run(test_y=eval_data[const.LABEL_COLUMN], y_prop=eval_prop,
                                              logger=self.logger)
        title_tables.append(("eval_data_metric", eval_metric_output["ks_bucket"]))

        utils.pr_plot(test_y=eval_data[const.LABEL_COLUMN], y_prop=eval_prop, label='test', title_images=title_images)
        utils.dist_plot(test_y=eval_data[const.LABEL_COLUMN], y_prop=eval_prop, title_images=title_images)

        utils.create_shap_column(model=xgb_model, predict_data=eval_data, feature_columns=train_feature_columns)

        train_data[const.PREDICT_COLUMN] = train_prop
        eval_data[const.PREDICT_COLUMN] = eval_prop

        ## 存储
        self.logger.info("xgboost save")
        DATAS_DIR = os.path.abspath("datas")
        DATAS_MODELS_DIR = f"{DATAS_DIR}/models"
        model_name = f"xgboost"
        model_path = f"{DATAS_MODELS_DIR}/{model_name}"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        xgb_model.save_model(model_path)
        self.set_model_path(model_path)

        # importance
        self.logger.info(f"feature_importances: {xgb_model.feature_importances_}")
        if np.all(xgb_model.feature_importances_ == 0):
            logging.warning(
                "all feature importances is 0, can not plot importance!")
        else:
            pyplot.figure()
            pyplot.rcParams["figure.figsize"] = (10, 20)
            xgboost.plot_importance(
                xgb_model,
                max_num_features=32,
                importance_type='gain')
            importance_image_url = utils.plot_show(
                pyplot, f'xgb_feature_importance_xgb_{self.base_info.graph_name}.png')
            title_images.append(("feature_importance", importance_image_url))

        self.set_images(title_images)
        self.set_tables(title_tables)

        return {"model": xgb_model, "eval_data": eval_data}