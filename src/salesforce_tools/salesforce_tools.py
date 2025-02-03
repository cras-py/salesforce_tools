import pandas as pd
import matplotlib.pyplot as plt
from simple_salesforce import Salesforce
from datetime import datetime
import numpy as np

class SalesforceTools:
    def __init__(self, username, password, security_token, domain='login'):
        self.sf = Salesforce(username=username, password=password, security_token=security_token, domain=domain)
        self.opportunities = None
    
    def sf_api_query(self,soql, dateList=None, tz=None):
        data = self.sf.query_all(soql)
        try:
            df = pd.DataFrame(data["records"]).drop("attributes", axis=1)
            listColumns = list(df.columns)
            for col in listColumns:
                if any(
                    isinstance(df[col].values[i], dict)
                    for i in range(0, len(df[col].values))
                ):
                    df = pd.concat(
                        [
                            df.drop(columns=[col]),
                            df[col]
                            .apply(pd.Series)
                            .drop("attributes", axis=1)
                            .add_prefix(col + "."),
                        ],
                        axis=1,
                    )
                    new_columns = np.setdiff1d(df.columns, listColumns)
                    for i in new_columns:
                        listColumns.append(i)
            try:
                for date in dateList:
                    if max(df[date].str.len()) > 10:
                        try:
                            df[date] = (
                                pd.to_datetime(df[date])
                                .dt.tz_convert(tz)
                                .dt.tz_localize(None)
                            )
                        except:
                            pass
                    else:
                        try:
                            df[date] = pd.to_datetime(df[date])
                        except:
                            pass
            except:
                pass
            return df
        except:
            print("The Query returned 0 rows")
    
    def query_opportunities(self,dateList,tz):
        current_year = datetime.now().year
        last_two_years_start = f"{current_year - 2}-01-01"
        next_year_end = f"{current_year + 1}-12-31"

        soql_query = f"""
        SELECT Id, Name, StageName, Amount, CloseDate, CreatedDate, IsWon, IsClosed, OwnerId, Type, Probability, AccountId
        FROM Opportunity
        WHERE CloseDate >= {last_two_years_start} AND CloseDate <= {next_year_end}
        """
        self.opportunities = self.sf_api_query(soql_query,dateList,tz)
        return self.opportunities
    
    def calculate_kpis(self):
        if self.opportunities is None:
            raise ValueError("Opportunities not yet queried. Call query_opportunities() first.")
        
        df = self.opportunities.copy()
        df['CloseDate'] = pd.to_datetime(df['CloseDate']).dt.tz_localize(None)  # Remove timezone info
        df['CreatedDate'] = pd.to_datetime(df['CreatedDate']).dt.tz_localize(None)  # Remove timezone info
        df['Year'] = df['CloseDate'].dt.year

        total_opps = len(df)
        won_opps = df[df['IsWon'] == True]
        lost_opps = df[df['IsWon'] == False]
        closed_opps = df[df['IsClosed'] == True]
        open_opps = df[df['IsClosed'] == False]

        kpis = {
            # General KPIs
            "Total Opportunities": total_opps,
            "Total Won Opportunities": len(won_opps),
            "Total Lost Opportunities": len(lost_opps),
            "Total Closed Opportunities": len(closed_opps),
            "Total Open Opportunities": len(open_opps),
            "Win Rate (%)": (len(won_opps) / len(closed_opps) * 100) if len(closed_opps) > 0 else 0,
            "Loss Rate (%)": (len(lost_opps) / len(closed_opps) * 100) if len(closed_opps) > 0 else 0,
            "Pipeline Value": df['Amount'].sum(),
            "Average Deal Size": df['Amount'].mean(),
            
            # Conversion KPIs
            "Open-to-Close Conversion Rate (%)": (len(closed_opps) / total_opps * 100) if total_opps > 0 else 0,
            "Stage-to-Stage Conversion Rates": df.groupby('StageName')['IsClosed'].mean().to_dict(),
            "Win Probability Average (%)": df['Probability'].mean(),

            # Time-based KPIs
            "Average Sales Cycle (Days)": (closed_opps['CloseDate'] - closed_opps['CreatedDate']).dt.days.mean(),
            "Opportunities Created Per Month": total_opps / df['CreatedDate'].dt.month.nunique(),

            # Revenue KPIs
            "Total Revenue from Won Deals": won_opps['Amount'].sum(),
            "Average Revenue Per Deal": won_opps['Amount'].mean(),
            "Revenue Growth YoY (%)": self._calculate_revenue_growth(df),
            "Revenue from Expansion": won_opps[won_opps['Type'] == 'Expansion']['Amount'].sum(),
            "Revenue from New Business": won_opps[won_opps['Type'] == 'New Business']['Amount'].sum(),

            # Pipeline KPIs
            "Pipeline Growth Rate (%)": self._calculate_pipeline_growth(df),
            "Weighted Pipeline Value": (df['Amount'] * df['Probability']).sum(),
            "Pipeline Coverage Ratio": self._pipeline_coverage_ratio(df, won_opps),

            # Sales Productivity
            "Opportunities Per Sales Rep": df.groupby('OwnerId').size().mean(),
            "Won Opportunities Per Sales Rep": won_opps.groupby('OwnerId').size().mean(),
            "Revenue Per Sales Rep": won_opps.groupby('OwnerId')['Amount'].sum().mean(),
            "Average Revenue Per Month Per Rep": won_opps.groupby('OwnerId')['Amount'].sum().mean() / df['CreatedDate'].dt.month.nunique(),

            # Forecasting KPIs
            "Expected Revenue from Open Opportunities": open_opps['Amount'].sum() * 0.5,  # Assume 50% close rate
            "Quarterly Revenue Forecast": self._quarterly_forecast(won_opps),

            # SaaS-specific KPIs
            "Churned Revenue": lost_opps['Amount'].sum(),
            "ARR (Annual Recurring Revenue)": won_opps['Amount'].sum() / 12 * 12,
            "MRR (Monthly Recurring Revenue)": won_opps['Amount'].sum() / 12,
            "Net Revenue Retention (%)": self._net_revenue_retention(won_opps),
            "Gross Revenue Retention (%)": self._gross_revenue_retention(won_opps),
            "Expansion Revenue": won_opps[won_opps['Type'] == 'Expansion']['Amount'].sum(),
            "Upsell Revenue": self._calculate_upsell(won_opps),
            "Cross-Sell Revenue": self._calculate_cross_sell(won_opps),
            "Customer Lifetime Value (CLV)": self._customer_lifetime_value(won_opps),
            "Customer Acquisition Cost (CAC)": self._customer_acquisition_cost(won_opps),
            "CAC Payback Period (Months)": self._cac_payback_period(won_opps),

            # Growth Metrics
            "Revenue Growth Rate YoY (%)": self._calculate_revenue_growth(df),
            "Customer Growth Rate YoY (%)": self._customer_growth_rate(df),
            "Bookings Growth Rate (%)": self._calculate_bookings_growth(df),
            "Opportunity Velocity (Deals/Day)": len(df) / (df['CloseDate'].max() - df['CreatedDate'].min()).days,
        }

        return kpis
    
    def _calculate_revenue_growth(self, df):
        revenue_by_year = df[df['IsWon'] == True].groupby(df['Year'])['Amount'].sum()
        if len(revenue_by_year) < 2:
            return 0
        return ((revenue_by_year.iloc[-1] - revenue_by_year.iloc[-2]) / revenue_by_year.iloc[-2]) * 100
    
    def _calculate_bookings_growth(self, closed_opps):
        """
        Calculates the growth rate of closed bookings over time.
        """
        bookings_by_year = closed_opps.groupby(closed_opps['Year'])['Amount'].sum()
        if len(bookings_by_year) < 2:
            return 0
        return ((bookings_by_year.iloc[-1] - bookings_by_year.iloc[-2]) / bookings_by_year.iloc[-2]) * 100
    
    def _customer_growth_rate(self, df):
        """
        Calculates the year-over-year customer growth rate.
        """
        customers_by_year = df.groupby('Year')['AccountId'].nunique()  # Count unique AccountIds per year
        if len(customers_by_year) < 2:
            return 0
        return ((customers_by_year.iloc[-1] - customers_by_year.iloc[-2]) / customers_by_year.iloc[-2]) * 100

    def _calculate_pipeline_growth(self, df):
        pipeline_by_year = df.groupby(df['Year'])['Amount'].sum()
        if len(pipeline_by_year) < 2:
            return 0
        return ((pipeline_by_year.iloc[-1] - pipeline_by_year.iloc[-2]) / pipeline_by_year.iloc[-2]) * 100
    
    def _pipeline_coverage_ratio(self, df, won_opps):
        pipeline_value = df['Amount'].sum()
        closed_revenue = won_opps['Amount'].sum()
        return pipeline_value / closed_revenue if closed_revenue > 0 else 0

    def _quarterly_forecast(self, won_opps):
        return won_opps.groupby([won_opps['CloseDate'].dt.year,won_opps['CloseDate'].dt.quarter])['Amount'].sum().to_dict()
    
    def _net_revenue_retention(self, won_opps):
        return (won_opps['Amount'].sum() - won_opps[won_opps['Type'] == 'Churn']['Amount'].sum()) / won_opps['Amount'].sum() * 100

    def _gross_revenue_retention(self, won_opps):
        return won_opps[won_opps['Type'] != 'Churn']['Amount'].sum() / won_opps['Amount'].sum() * 100
    
    def _calculate_upsell(self, won_opps):
        return won_opps[won_opps['Type'] == 'Upsell']['Amount'].sum()
    
    def _calculate_cross_sell(self, won_opps):
        return won_opps[won_opps['Type'] == 'Cross-Sell']['Amount'].sum()
    
    def _customer_lifetime_value(self, won_opps):
        avg_revenue_per_customer = won_opps['Amount'].mean()
        churn_rate = len(won_opps) / len(won_opps['OwnerId'].unique())
        return avg_revenue_per_customer / churn_rate if churn_rate > 0 else 0
    
    def _customer_acquisition_cost(self, won_opps):
        # Placeholder: Adjust based on marketing/sales cost data
        sales_and_marketing_cost = 50000
        return sales_and_marketing_cost / len(won_opps) if len(won_opps) > 0 else 0

    def _cac_payback_period(self, won_opps):
        cac = self._customer_acquisition_cost(won_opps)
        mrr = self._calculate_mrr(won_opps)
        return cac / mrr if mrr > 0 else 0

    def _calculate_mrr(self, won_opps):
        return won_opps['Amount'].sum() / 12 if len(won_opps) > 0 else 0
