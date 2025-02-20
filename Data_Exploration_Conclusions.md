# Report

All the base for this report is found in the `artifacts/exploratory_data_analysys` folder. Please, make sure to check this folder

| FEATURE                          | DESCRIPTION                                                                                                                     | CLASSIFICATION |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| CUST_ID                          | Identification of Credit Card holder (Categorical)                                                                              | id             |
| BALANCE                          | Balance amount left in their account to make purchases                                                                          | numeric        |
| BALANCE_FREQUENCY                | How frequently the Balance is updated score between 0 and 1<br />(1 = frequently updated 0 = not frequently updated)           | numeric        |
| PURCHASES                        | Amount of purchases made from account                                                                                           | numeric        |
| ONEOFF_PURCHASES                 | Maximum purchase amount done in one-go                                                                                          | numeric        |
| INSTALLMENTS_PURCHASES           | Amount of purchase done in installment                                                                                          | numeric        |
| CASH_ADVANCE                     | Cash in advance given by the user                                                                                               | numeric        |
| PURCHASES_FREQUENCY              | How frequently the Purchases are being made score<br />between 0 and 1 (1 = frequently purchased 0 = not frequently purchased) | numeric        |
| ONEOFF_PURCHASES_FREQUENCY       | How frequently Purchases are happening in one-go<br />(1 = frequently purchased 0 = not frequently purchased)                  | numeric        |
| PURCHASES_INSTALLMENTS_FREQUENCY | How frequently purchases in installments are being done<br />(1 = frequently done 0 = not frequently done)                     | numeric        |
| CASH_ADVANCE_FREQUENCY           | How frequently the cash in advance being paid                                                                                   | numeric        |
| CASH_ADVANCE_TRX                 | Number of Transactions made with "Cash in Advanced"                                                                             | numeric        |
| PURCHASES_TRX                    | Numbe of purchase transactions made                                                                                             | numeric        |
| CREDIT_LIMIT                     | Limit of Credit Card for user                                                                                                   | numeric        |
| PAYMENTS                         | Amount of Payment done by user                                                                                                  | numeric        |
| MINIMUM_PAYMENTS                 | Minimum amount of payments made by user                                                                                         | numeric        |
| PRC_FULL_PAYMENT                 | Percent of full payment paid by user                                                                                            | numeric        |
| TENURE                           | Tenure of credit card service for user                                                                                          | numeric        |

https://www.kaggle.com/datasets/arjunbhasin2013/ccdata

Dataframe dimensions 18 x 8950

## Data Quality

* `CREDIT_LIMIT` has 1 missing value **[1]**;
* `MINIMUM_PAYMENTS` has 313 missing values (3.5%) **[2]**;
* There are many collumns with outliers **[3]**

### Unique Value Count

| COLUMN                           | NunUnique |
| :------------------------------- | --------: |
| CUST_ID                          |      8950 |
| BALANCE                          |      8871 |
| BALANCE_FREQUENCY                |        43 |
| PURCHASES                        |      6203 |
| ONEOFF_PURCHASES                 |      4014 |
| INSTALLMENTS_PURCHASES           |      4452 |
| CASH_ADVANCE                     |      4323 |
| PURCHASES_FREQUENCY              |        47 |
| ONEOFF_PURCHASES_FREQUENCY       |        47 |
| PURCHASES_INSTALLMENTS_FREQUENCY |        47 |
| CASH_ADVANCE_FREQUENCY           |        54 |
| CASH_ADVANCE_TRX                 |        65 |
| PURCHASES_TRX                    |       173 |
| CREDIT_LIMIT                     |       205 |
| PAYMENTS                         |      8711 |
| MINIMUM_PAYMENTS                 |      8636 |
| PRC_FULL_PAYMENT                 |        47 |
| TENURE                           |         7 |

## Data Distribution

### Univariate

* All the columns have several outliers, except: `PURCHASES_FREQUENCY`  `PURCHASES_INSTALLMENTS_FREQUENCY`   **[3];**
* There are data within different scales **[4]**

#### Distributions [5]

Right Skewed: `BALANCE`, `PURCHASES`, `ONEOFF_PURCHASES`, `INSTALLMENTS_PURCHASES`, `CASH_ADVANCE`,`ONEOFF_PURCHASES_FREQUENCY`, `CASH_ADVANCE_FREQUENCY`, `CASH_ADVANCE_TRX`, `PURCHASES_TRX`, `CREDIT_LIMIT`, `PAYMENTS`, `MINIMUM_PAYMENTS`, `PRC_FULL_PAYMENT`

Left Skewed: `BALANCE_FREQUENCY`, `TENURE`

Bimodal Distribution: `PURCHASES_FREQUENCY`, `PURCHASES_INSTALLMENTS_FREQUENCY`

### Bi-Variate

* Nothing much important,s most of relationships are described in correlations.

### Correlations

* There are several columns that are highly correlated between themselves **[6]**:

![1738804324810](image/Data_Exploration_Conclusions/1738804324810.png)

## Conclusions

* A

## Processment Ideas

* []
