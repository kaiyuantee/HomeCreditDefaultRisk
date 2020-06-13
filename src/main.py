import gc
import pandas as pd
from .utils import timer, reduce_memory
from .ratio_feats import add_ratios_features
from .model import kfold_lightgbm_sklearn
from .application_pipeline import get_train_test
from .bureau_pipeline import get_bureau
from .previou_pipeline import get_previous_applications
from .pos_cash_pipeline import get_pos_cash
from .installments_pipeline import get_installment_payments
from .creditcard_pipeline import get_credit_card
from .cfgs import DATA_DIRECTORY
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def main(debug=False):
    num_rows = 30000 if debug else None
    with timer('application_train and application_test'):
        df = get_train_test(DATA_DIRECTORY, num_rows=num_rows)
        print('Application dataframe shape:', df.shape)
    with timer("Bureau and bureau_balance data"):
        bureau_df = get_bureau(DATA_DIRECTORY, num_rows=num_rows)
        df = pd.merge(df, bureau_df, on='SK_ID_CURR', how='left')
        print("Bureau dataframe shape: ", bureau_df.shape)
        del bureau_df;
        gc.collect()
    with timer("previous_application"):
        prev_df = get_previous_applications(DATA_DIRECTORY, num_rows)
        df = pd.merge(df, prev_df, on='SK_ID_CURR', how='left')
        print("Previous dataframe shape: ", prev_df.shape)
        del prev_df;
        gc.collect()
    with timer("previous applications balances"):
        pos = get_pos_cash(DATA_DIRECTORY, num_rows)
        df = pd.merge(df, pos, on='SK_ID_CURR', how='left')
        print("Pos-cash dataframe shape: ", pos.shape)
        del pos;
        gc.collect()
        ins = get_installment_payments(DATA_DIRECTORY, num_rows)
        df = pd.merge(df, ins, on='SK_ID_CURR', how='left')
        print("Installments dataframe shape: ", ins.shape)
        del ins;
        gc.collect()
        cc = get_credit_card(DATA_DIRECTORY, num_rows)
        df = pd.merge(df, cc, on='SK_ID_CURR', how='left')
        print("Credit card dataframe shape: ", cc.shape)
        del cc;
        gc.collect()

        # Add ratios and groupby between different tables
        df = add_ratios_features(df)
        df = reduce_memory(df)
        lgbm_categorical_feat = ['CODE_GENDER', 'FLAG_OWN_CAR', 'NAME_CONTRACT_TYPE', 'NAME_EDUCATION_TYPE',
                                 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'NAME_INCOME_TYPE', 'OCCUPATION_TYPE',
                                 'ORGANIZATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'NAME_TYPE_SUITE', 'WALLSMATERIAL_MODE']
        with timer('Running LGBM Now'):
            feat_importance = kfold_lightgbm_sklearn(df, lgbm_categorical_feat)
            print(feat_importance)


if __name__ == "__main__":
    pd.set_option('display.max_rows', 60)
    pd.set_option('display.max_columns', 100)
    with timer("Pipeline total time"):
        main(debug= False)