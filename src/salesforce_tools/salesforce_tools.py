from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from simple_salesforce import Salesforce
from datetime import datetime
import logging
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_START_DATE = f"{datetime.now().year}-01-01"
DEFAULT_END_DATE = f"{datetime.now().year}-12-31"
DEFAULT_WIN_PROBABILITY = 0.5
DEFAULT_MARKETING_COST = 50000  # Should be configured based on actual costs


class SalesforceTools:
    """
    A class for retrieving and analyzing Salesforce opportunity data.
    
    This class provides methods for querying Salesforce opportunities and 
    calculating various sales KPIs (Key Performance Indicators).
    
    Attributes:
        sf (Salesforce): The Salesforce API connection object.
        opportunities (pd.DataFrame): DataFrame containing opportunity data.
    """

    def __init__(self, username: str, password: str, security_token: str, domain: str = 'login'):
        """
        Initialize the SalesforceTools with authentication parameters.
        
        Args:
            username: Salesforce username
            password: Salesforce password
            security_token: Salesforce security token
            domain: Salesforce domain (default: 'login')
        """
        self.sf = Salesforce(username=username, password=password, security_token=security_token, domain=domain)
        self.opportunities = None
    
    def sf_api_query(self, soql: str, date_columns: Optional[List[str]] = None, 
                    timezone: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Execute a SOQL query against Salesforce and transform the results into a pandas DataFrame.
        
        Args:
            soql: A valid SOQL query string
            date_columns: List of column names containing date values to convert to datetime
            timezone: Timezone to convert the dates to
            
        Returns:
            DataFrame containing the query results or None if the query returns no rows
            
        Example:
            >>> tools = SalesforceTools(username, password, token)
            >>> query = "SELECT Id, Name FROM Account LIMIT 10"
            >>> df = tools.sf_api_query(query)
        """
        try:
            # Execute the query
            data = self.sf.query_all(soql)
            
            # Handle empty results
            if not data.get("records"):
                logger.warning("The query returned 0 rows")
                return None
                
            # Transform into DataFrame
            df = pd.DataFrame(data["records"]).drop("attributes", axis=1)
            
            # Process nested objects
            df = self._flatten_nested_objects(df)
                
            # Convert date columns if specified
            if date_columns:
                df = self._convert_date_columns(df, date_columns, timezone)
                
            return df
            
        except Exception as e:
            logger.error(f"Error executing SOQL query: {str(e)}")
            raise
    
    def _flatten_nested_objects(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flattens nested objects in the DataFrame into separate columns.
        
        Args:
            df: DataFrame potentially containing nested JSON objects
            
        Returns:
            Flattened DataFrame
        """
        list_columns = list(df.columns)
        for col in list_columns:
            # Check if column contains dictionary values
            if any(isinstance(df[col].values[i], dict) for i in range(0, len(df[col].values))):
                # Extract nested data
                nested_df = df[col].apply(pd.Series).drop("attributes", axis=1).add_prefix(col + ".")
                
                # Combine with original DataFrame
                df = pd.concat([df.drop(columns=[col]), nested_df], axis=1)
                
                # Update the list of columns
                new_columns = np.setdiff1d(df.columns, list_columns)
                list_columns.extend(new_columns)
                
        return df
    
    def flatten_dict(self, d, parent_key='', sep=' - '):
        """
        Recursively flattens a nested dictionary.
        Nested keys are joined with the provided separator.
        """
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self.flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items
    
    def _convert_date_columns(self, df: pd.DataFrame, date_columns: List[str], 
                             timezone: Optional[str] = None) -> pd.DataFrame:
        """
        Convert columns containing date strings to datetime objects.
        
        Args:
            df: DataFrame containing date columns
            date_columns: List of column names to convert
            timezone: Optional timezone for conversion
            
        Returns:
            DataFrame with converted date columns
        """
        for date_col in date_columns:
            if date_col not in df.columns:
                continue
                
            try:
                # Handle datetime with timezone
                if df[date_col].str.len().max() > 10:
                    try:
                        if timezone:
                            df[date_col] = (pd.to_datetime(df[date_col])
                                          .dt.tz_convert(timezone)
                                          .dt.tz_localize(None))
                        else:
                            df[date_col] = pd.to_datetime(df[date_col])
                    except Exception as e:
                        logger.warning(f"Failed to convert {date_col} with timezone: {str(e)}")
                # Handle date only
                else:
                    df[date_col] = pd.to_datetime(df[date_col])
            except Exception as e:
                logger.warning(f"Failed to convert {date_col} to datetime: {str(e)}")
                
        return df
    
    def query_opportunities(self, date_columns: Optional[List[str]] = None, 
                           timezone: Optional[str] = None,
                           start_date: Optional[str] = None, 
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Query opportunity data from Salesforce for a specified date range.
        
        Args:
            date_columns: List of date column names to convert to datetime
            timezone: Timezone for date conversion
            start_date: Start date for filtering opportunities (format: 'YYYY-MM-DD')
            end_date: End date for filtering opportunities (format: 'YYYY-MM-DD')
            
        Returns:
            DataFrame containing opportunity data
            
        Example:
            >>> tools = SalesforceTools(username, password, token)
            >>> opps = tools.query_opportunities(
            ...     date_columns=['CloseDate', 'CreatedDate'],
            ...     start_date='2023-01-01',
            ...     end_date='2023-12-31'
            ... )
        """
        # Use default date range if not specified
        if start_date is None:
            start_date = DEFAULT_START_DATE
        if end_date is None:
            end_date = DEFAULT_END_DATE
            
        # Default date columns if not specified
        if date_columns is None:
            date_columns = ['CloseDate', 'CreatedDate']

        timezone='America/Chicago'
            
        # Build the SOQL query
        soql_query = f"""
        SELECT Id, Name, StageName, Amount, CloseDate, CreatedDate, 
               IsWon, IsClosed, OwnerId, Type, Probability, AccountId
        FROM Opportunity
        WHERE CloseDate >= {start_date} AND CloseDate <= {end_date}
        """
        
        # Execute the query
        self.opportunities = self.sf_api_query(soql_query, date_columns, timezone)
        
        if self.opportunities is None:
            # Return empty DataFrame with expected columns if no results
            columns = ['Id', 'Name', 'StageName', 'Amount', 'CloseDate', 'CreatedDate',
                      'IsWon', 'IsClosed', 'OwnerId', 'Type', 'Probability', 'AccountId']
            self.opportunities = pd.DataFrame(columns=columns)
            
        return self.opportunities
    
    def calculate_kpis(self) -> Dict[str, Any]:
        """
        Calculate key performance indicators (KPIs) based on queried opportunity data.
        
        Returns:
            Dictionary containing calculated KPIs
            
        Raises:
            ValueError: If opportunities have not been queried yet
            
        Example:
            >>> tools = SalesforceTools(username, password, token)
            >>> tools.query_opportunities()
            >>> kpis = tools.calculate_kpis()
            >>> win_rate = kpis["Win Rate (%)"]
        """
        if self.opportunities is None or len(self.opportunities) == 0:
            raise ValueError("Opportunities not yet queried or empty. Call query_opportunities() first.")
        
        # Create a copy to avoid modifying the original data
        df = self.opportunities.copy()
        
        # Ensure date columns are datetime objects without timezone info
        for date_col in ['CloseDate', 'CreatedDate']:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
        
        # Add year column for time-based analysis
        df['Year'] = df['CloseDate'].dt.year
        df['Quarter'] = df['CloseDate'].dt.quarter
        
        # Filter opportunity segments
        segments = self._segment_opportunities(df)
        
        # Calculate basic metrics
        basic_metrics = self._calculate_basic_metrics(df, segments)
        
        # Calculate conversion metrics
        conversion_metrics = self._calculate_conversion_metrics(df, segments)
        
        # Calculate time-based metrics
        time_metrics = self._calculate_time_metrics(df, segments)
        
        # Calculate revenue metrics
        revenue_metrics = self._calculate_revenue_metrics(df, segments)
        
        # Calculate pipeline metrics
        pipeline_metrics = self._calculate_pipeline_metrics(df, segments)
        
        # Calculate sales productivity metrics
        productivity_metrics = self._calculate_productivity_metrics(df, segments)
        
        # Calculate forecasting metrics
        forecasting_metrics = self._calculate_forecasting_metrics(df, segments)
        
        # Calculate SaaS-specific metrics
        saas_metrics = self._calculate_saas_metrics(df, segments)
        
        # Calculate growth metrics
        growth_metrics = self._calculate_growth_metrics(df, segments)
        
        # Combine all metrics
        kpis = {
            **basic_metrics,
            **conversion_metrics,
            **time_metrics,
            **revenue_metrics,
            **pipeline_metrics,
            **productivity_metrics,
            **forecasting_metrics,
            **saas_metrics,
            **growth_metrics
        }
        
        return kpis
    
    def _segment_opportunities(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Segment opportunities into different categories for analysis.
        
        Args:
            df: DataFrame containing opportunity data
            
        Returns:
            Dictionary of DataFrames for different segments
        """
        segments = {
            "total": df,
            "won": df[df['IsWon'] == True],
            "lost": df[df['IsClosed'] == True][df['IsWon'] == False],
            "closed": df[df['IsClosed'] == True],
            "open": df[df['IsClosed'] == False],
            "expansion": df[df['Type'] == 'Expansion'],
            "new_business": df[df['Type'] == 'New Business'],
            "upsell": df[df['Type'] == 'Upsell'],
            "cross_sell": df[df['Type'] == 'Cross-Sell'],
            "churn": df[df['Type'] == 'Churn']
        }
        
        # Add won versions of each segment
        segments.update({
            f"won_{key}": value[value['IsWon'] == True] 
            for key, value in segments.items() 
            if key not in ['won', 'lost', 'closed', 'open']
        })
        
        return segments
    
    def _calculate_basic_metrics(self, df: pd.DataFrame, 
                               segments: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate basic opportunity metrics."""
        total_opps = len(segments["total"])
        closed_opps = len(segments["closed"])
        
        metrics = {
            "Total Opportunities": total_opps,
            "Total Won Opportunities": len(segments["won"]),
            "Total Lost Opportunities": len(segments["lost"]),
            "Total Closed Opportunities": closed_opps,
            "Total Open Opportunities": len(segments["open"]),
            "Win Rate (%)": (len(segments["won"]) / closed_opps * 100) if closed_opps > 0 else 0,
            "Loss Rate (%)": (len(segments["lost"]) / closed_opps * 100) if closed_opps > 0 else 0,
        }
        
        return metrics
    
    def _calculate_conversion_metrics(self, df: pd.DataFrame, 
                                    segments: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate conversion-related metrics."""
        total_opps = len(segments["total"])
        closed_opps = len(segments["closed"])
        
        metrics = {
            "Open-to-Close Conversion Rate (%)": (closed_opps / total_opps * 100) if total_opps > 0 else 0,
            "Stage-to-Stage Conversion Rates": df.groupby('StageName')['IsClosed'].mean().to_dict(),
            "Win Probability Average (%)": df['Probability'].mean(),
        }
        
        return metrics
    
    def _calculate_time_metrics(self, df: pd.DataFrame, 
                              segments: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate time-based metrics."""
        closed_opps = segments["closed"]
        
        # Calculate average sales cycle
        if 'CreatedDate' in closed_opps.columns and 'CloseDate' in closed_opps.columns and len(closed_opps) > 0:
            avg_sales_cycle = (closed_opps['CloseDate'] - closed_opps['CreatedDate']).dt.days.mean()
        else:
            avg_sales_cycle = 0
            
        # Calculate opportunities per month
        if 'CreatedDate' in df.columns and len(df) > 0:
            month_count = df['CreatedDate'].dt.month.nunique()
            opps_per_month = len(df) / month_count if month_count > 0 else 0
        else:
            opps_per_month = 0
            
        metrics = {
            "Average Sales Cycle (Days)": avg_sales_cycle,
            "Opportunities Created Per Month": opps_per_month,
        }
        
        return metrics
    
    def _calculate_revenue_metrics(self, df: pd.DataFrame, 
                                 segments: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate revenue-related metrics."""
        won_opps = segments["won"]
        
        metrics = {
            "Total Revenue from Won Deals": won_opps['Amount'].sum() if 'Amount' in won_opps.columns else 0,
            "Average Revenue Per Deal": won_opps['Amount'].mean() if 'Amount' in won_opps.columns and len(won_opps) > 0 else 0,
            "Revenue Growth YoY (%)": self._calculate_revenue_growth(df) if 'Amount' in df.columns else 0,
            "Revenue from Expansion": segments["won_expansion"]['Amount'].sum() if 'Amount' in segments["won_expansion"].columns else 0,
            "Revenue from New Business": segments["won_new_business"]['Amount'].sum() if 'Amount' in segments["won_new_business"].columns else 0,
        }
        
        return metrics
    
    def _calculate_pipeline_metrics(self, df: pd.DataFrame, 
                                  segments: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate pipeline-related metrics."""
        won_opps = segments["won"]
        
        metrics = {
            "Pipeline Value": df['Amount'].sum() if 'Amount' in df.columns else 0,
            "Pipeline Growth Rate (%)": self._calculate_pipeline_growth(df) if 'Amount' in df.columns else 0,
            "Weighted Pipeline Value": (df['Amount'] * df['Probability'] / 100).sum() 
                                     if ('Amount' in df.columns and 'Probability' in df.columns) else 0,
            "Pipeline Coverage Ratio": self._pipeline_coverage_ratio(df, won_opps) 
                                     if ('Amount' in df.columns and 'Amount' in won_opps.columns) else 0,
        }
        
        return metrics
    
    def _calculate_productivity_metrics(self, df: pd.DataFrame, 
                                      segments: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate sales productivity metrics."""
        won_opps = segments["won"]
        
        metrics = {}
        
        # Only calculate if OwnerId is present
        if 'OwnerId' in df.columns:
            metrics.update({
                "Opportunities Per Sales Rep": df.groupby('OwnerId').size().mean() if len(df) > 0 else 0,
                "Won Opportunities Per Sales Rep": won_opps.groupby('OwnerId').size().mean() if len(won_opps) > 0 else 0,
            })
            
        # Only calculate if Amount is present
        if 'Amount' in won_opps.columns and 'OwnerId' in won_opps.columns:
            revenue_per_rep = won_opps.groupby('OwnerId')['Amount'].sum().mean() if len(won_opps) > 0 else 0
            metrics.update({
                "Revenue Per Sales Rep": revenue_per_rep,
            })
            
            # Add monthly metric if CreatedDate is available
            if 'CreatedDate' in df.columns:
                month_count = df['CreatedDate'].dt.month.nunique()
                metrics.update({
                    "Average Revenue Per Month Per Rep": revenue_per_rep / month_count if month_count > 0 else 0,
                })
        
        return metrics
    
    def _calculate_forecasting_metrics(self, df: pd.DataFrame, 
                                     segments: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate forecasting metrics."""
        open_opps = segments["open"]
        won_opps = segments["won"]
        
        metrics = {
            "Expected Revenue from Open Opportunities": open_opps['Amount'].sum() * DEFAULT_WIN_PROBABILITY 
                                                     if 'Amount' in open_opps.columns else 0,
        }
        
        # Add quarterly forecast if CloseDate is available
        if 'CloseDate' in won_opps.columns and 'Amount' in won_opps.columns:
            quarter_years = won_opps.groupby([won_opps['CloseDate'].dt.year, won_opps['CloseDate'].dt.quarter])
            metrics.update({
                "Quarterly Revenue Forecast": quarter_years['Amount'].sum().to_dict(),
            })
        
        return metrics
    
    def _calculate_saas_metrics(self, df: pd.DataFrame, 
                              segments: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate SaaS-specific metrics."""
        won_opps = segments["won"]
        lost_opps = segments["lost"]
        
        metrics = {}
        
        # Only calculate if Amount is present
        if 'Amount' in lost_opps.columns:
            metrics.update({
                "Churned Revenue": lost_opps['Amount'].sum(),
            })
            
        if 'Amount' in won_opps.columns:
            annual_revenue = won_opps['Amount'].sum()
            metrics.update({
                "ARR (Annual Recurring Revenue)": annual_revenue,
                "MRR (Monthly Recurring Revenue)": annual_revenue / 12,
            })
            
        # Add metrics that depend on Type field
        if 'Amount' in won_opps.columns and 'Type' in won_opps.columns:
            churn_amount = segments["won_churn"]['Amount'].sum() if 'Amount' in segments["won_churn"].columns else 0
            total_amount = won_opps['Amount'].sum()
            
            if total_amount > 0:
                net_retention = ((total_amount - churn_amount) / total_amount) * 100
                gross_retention = (segments["won"][segments["won"]['Type'] != 'Churn']['Amount'].sum() / total_amount) * 100
            else:
                net_retention = 0
                gross_retention = 0
                
            metrics.update({
                "Net Revenue Retention (%)": net_retention,
                "Gross Revenue Retention (%)": gross_retention,
                "Expansion Revenue": segments["won_expansion"]['Amount'].sum() if 'Amount' in segments["won_expansion"].columns else 0,
                "Upsell Revenue": segments["won_upsell"]['Amount'].sum() if 'Amount' in segments["won_upsell"].columns else 0,
                "Cross-Sell Revenue": segments["won_cross_sell"]['Amount'].sum() if 'Amount' in segments["won_cross_sell"].columns else 0,
            })
            
        # Calculate complex metrics
        metrics.update({
            "Customer Lifetime Value (CLV)": self._customer_lifetime_value(won_opps),
            "Customer Acquisition Cost (CAC)": self._customer_acquisition_cost(won_opps),
            "CAC Payback Period (Months)": self._cac_payback_period(won_opps),
        })
        
        return metrics
    
    def _calculate_growth_metrics(self, df: pd.DataFrame, 
                                segments: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate growth-related metrics."""
        metrics = {
            "Revenue Growth Rate YoY (%)": self._calculate_revenue_growth(df),
            "Customer Growth Rate YoY (%)": self._customer_growth_rate(df),
            "Bookings Growth Rate (%)": self._calculate_bookings_growth(segments["closed"]),
        }
        
        # Calculate opportunity velocity if dates are available
        if 'CloseDate' in df.columns and 'CreatedDate' in df.columns and len(df) > 0:
            date_range_days = (df['CloseDate'].max() - df['CreatedDate'].min()).days
            if date_range_days > 0:
                metrics.update({
                    "Opportunity Velocity (Deals/Day)": len(df) / date_range_days,
                })
        
        return metrics
    
    def _calculate_revenue_growth(self, df: pd.DataFrame) -> float:
        """
        Calculate year-over-year revenue growth from won opportunities.
        
        Args:
            df: DataFrame containing opportunity data
            
        Returns:
            Growth percentage or 0 if insufficient data
        """
        if 'Year' not in df.columns or 'Amount' not in df.columns or 'IsWon' not in df.columns:
            return 0
            
        revenue_by_year = df[df['IsWon'] == True].groupby('Year')['Amount'].sum()
        
        if len(revenue_by_year) < 2:
            return 0
            
        return ((revenue_by_year.iloc[-1] - revenue_by_year.iloc[-2]) / revenue_by_year.iloc[-2]) * 100
    
    def _calculate_bookings_growth(self, closed_opps: pd.DataFrame) -> float:
        """
        Calculate year-over-year growth rate of closed bookings.
        
        Args:
            closed_opps: DataFrame containing closed opportunity data
            
        Returns:
            Growth percentage or 0 if insufficient data
        """
        if 'Year' not in closed_opps.columns or 'Amount' not in closed_opps.columns:
            return 0
            
        bookings_by_year = closed_opps.groupby('Year')['Amount'].sum()
        
        if len(bookings_by_year) < 2:
            return 0
            
        return ((bookings_by_year.iloc[-1] - bookings_by_year.iloc[-2]) / bookings_by_year.iloc[-2]) * 100
    
    def _customer_growth_rate(self, df: pd.DataFrame) -> float:
        """
        Calculate year-over-year customer growth rate.
        
        Args:
            df: DataFrame containing opportunity data
            
        Returns:
            Growth percentage or 0 if insufficient data
        """
        if 'Year' not in df.columns or 'AccountId' not in df.columns:
            return 0
            
        customers_by_year = df.groupby('Year')['AccountId'].nunique()
        
        if len(customers_by_year) < 2:
            return 0
            
        return ((customers_by_year.iloc[-1] - customers_by_year.iloc[-2]) / customers_by_year.iloc[-2]) * 100

    def _calculate_pipeline_growth(self, df: pd.DataFrame) -> float:
        """
        Calculate year-over-year pipeline growth.
        
        Args:
            df: DataFrame containing opportunity data
            
        Returns:
            Growth percentage or 0 if insufficient data
        """
        if 'Year' not in df.columns or 'Amount' not in df.columns:
            return 0
            
        pipeline_by_year = df.groupby('Year')['Amount'].sum()
        
        if len(pipeline_by_year) < 2:
            return 0
            
        return ((pipeline_by_year.iloc[-1] - pipeline_by_year.iloc[-2]) / pipeline_by_year.iloc[-2]) * 100
    
    def _pipeline_coverage_ratio(self, df: pd.DataFrame, won_opps: pd.DataFrame) -> float:
        """
        Calculate the ratio of total pipeline value to closed revenue.
        
        Args:
            df: DataFrame containing all opportunity data
            won_opps: DataFrame containing won opportunity data
            
        Returns:
            Coverage ratio or 0 if insufficient data
        """
        if 'Amount' not in df.columns or 'Amount' not in won_opps.columns:
            return 0
            
        pipeline_value = df['Amount'].sum()
        closed_revenue = won_opps['Amount'].sum()
        
        return pipeline_value / closed_revenue if closed_revenue > 0 else 0
    
    def _customer_lifetime_value(self, won_opps: pd.DataFrame) -> float:
        """
        Estimate the customer lifetime value based on average revenue and churn.
        
        Args:
            won_opps: DataFrame containing won opportunity data
            
        Returns:
            Estimated CLV or 0 if insufficient data
        """
        if 'Amount' not in won_opps.columns or 'OwnerId' not in won_opps.columns or len(won_opps) == 0:
            return 0
            
        avg_revenue_per_customer = won_opps['Amount'].mean()
        
        # Use a simplified approximation of churn rate
        unique_owners = len(won_opps['OwnerId'].unique())
        churn_rate = len(won_opps) / unique_owners if unique_owners > 0 else 0
        
        return avg_revenue_per_customer / churn_rate if churn_rate > 0 else 0
    
    def _customer_acquisition_cost(self, won_opps: pd.DataFrame) -> float:
        """
        Calculate an approximated customer acquisition cost.
        
        Args:
            won_opps: DataFrame containing won opportunity data
                
        Returns:
            Estimated CAC or 0 if insufficient data
        """
        # Verify input type to avoid cryptic errors
        if not isinstance(won_opps, pd.DataFrame):
            logger.warning(f"Expected DataFrame for won_opps, got {type(won_opps)}")
            return 0
            
        if len(won_opps) == 0:
            return 0
            
        # Use the global constant directly
        return DEFAULT_MARKETING_COST / len(won_opps)
    
    def _cac_payback_period(self, won_opps: pd.DataFrame) -> float:
        """
        Calculate the estimated time to recover customer acquisition costs.
        
        Args:
            won_opps: DataFrame containing won opportunity data
            
        Returns:
            Estimated payback period in months or 0 if insufficient data
        """
        cac = self._customer_acquisition_cost(won_opps)
        
        if 'Amount' not in won_opps.columns or len(won_opps) == 0:
            return 0
            
        mrr = won_opps['Amount'].sum() / 12
        
        return cac / mrr if mrr > 0 else 0
    
    def visualize_pipeline(self, title: str = "Sales Pipeline by Stage", 
                          figsize: Tuple[int, int] = (12, 6)):
        """
        Create a visualization of the sales pipeline by stage.
        
        Args:
            title: The title for the plot
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Matplotlib figure
            
        Example:
            >>> tools = SalesforceTools(username, password, token)
            >>> tools.query_opportunities()
            >>> fig = tools.visualize_pipeline()
            >>> fig.savefig('pipeline.png')
        """
        import matplotlib.pyplot as plt
        
        if self.opportunities is None:
            raise ValueError("Opportunities not yet queried. Call query_opportunities() first.")
            
        df = self.opportunities.copy()
        
        # Group by stage and sum amounts
        pipeline_by_stage = df.groupby('StageName')['Amount'].sum().sort_values(ascending=False)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(pipeline_by_stage.index, pipeline_by_stage.values)
        
        # Add data labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + (width * 0.01), 
                   bar.get_y() + bar.get_height()/2, 
                   f'${width:,.0f}', 
                   va='center')
        
        # Add formatting
        ax.set_title(title)
        ax.set_xlabel('Amount ($)')
        ax.set_ylabel('Stage')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Format x-axis as currency
        from matplotlib.ticker import FuncFormatter
        def currency_formatter(x, pos):
            return f'${x:,.0f}'
        ax.xaxis.set_major_formatter(FuncFormatter(currency_formatter))
        
        plt.tight_layout()
        return fig
    
    def export_to_csv(self, df: pd.DataFrame, filename: str) -> None:
        """
        Export the opportunities DataFrame to a CSV file.
        
        Args:
            filename: Path to save the CSV file
            
        Raises:
            ValueError: If opportunities have not been queried yet
            
        Example:
            >>> tools = SalesforceTools(username, password, token)
            >>> tools.query_opportunities()
            >>> tools.export_to_csv('opportunities.csv')
        """
        if self.df is None:
            raise ValueError("df not yet created. Create df first.")
        
        try:
            df.to_csv(filename, index=False)
            logger.info(f"Successfully exported data to {filename}")
        except Exception as e:
            logger.error(f"Failed to export data to {filename}: {str(e)}")
            raise
    
    def export_to_excel(self, df: pd.DataFrame ,filename: str, include_kpis: bool = True) -> None:
        """
        Export the opportunities DataFrame to an Excel file.
        
        Args:
            filename: Path to save the Excel file
            include_kpis: Whether to include KPI calculations in the export (default: True)
                
        Raises:
            ValueError: If opportunities have not been queried yet
                
        Example:
            >>> tools = SalesforceTools(username, password, token)
            >>> tools.query_opportunities()
            >>> tools.export_to_excel('opportunities.xlsx')
        """
        
        # Create a copy of the DataFrame to avoid modifying the original
        export_df = df.copy()
        
        # Add KPI calculations if requested
        if include_kpis and hasattr(self, 'calculate_kpis'):
            # Flatten the dictionary
                kpi_df = self.flatten_dict(sf.calculate_kpis())

                # Convert the flattened dictionary to a DataFrame (one row per record)
                kpi_df = pd.DataFrame([kpi_df])
                kpi_df = kpi_df.melt(var_name='KPI', value_name='Value')
        
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            workbook = writer.book
            worksheet = workbook.add_worksheet("df")
            writer.sheets["df"] = worksheet

            # Define formats
            custom_date_format = workbook.add_format({'bold': True, 'num_format': 'mmm yyyy', 'align': 'center'})
            custom_date_format_rate = workbook.add_format({'bold': False, 'num_format': 'mmm yyyy', 'align': 'left'})
            category_format = workbook.add_format({'bold': True, 'align': 'center'})
            header_format = workbook.add_format({'bold': True, 'align': 'center', 'border': 1})

            # Apply filter to the second row
            worksheet.autofilter(0, 0, 1, len(export_df.columns)-1)

            # Freeze the top 2 rows and first 7 columns
            worksheet.freeze_panes(1,0)

            # Optionally, set column widths for better readability
            for i, col in enumerate(export_df.columns):
                # Calculate the max length of the column's data as strings, and compare it with the header's length
                max_length = max(export_df[col].astype(str).map(len).max(), len(col))
                # Set the column width to the maximum length plus a little extra for spacing
                worksheet.set_column(i, i, max_length + 2)

            export_df.to_excel(writer, sheet_name='df', index=False)
            
            # Add the second sheet with the same name as the model
            rates_sheet = workbook.add_worksheet('kpis')
            writer.sheets['kpis'] = rates_sheet
            for i, col in enumerate(kpi_df.columns):
                # Calculate the max length of the column's data as strings, and compare it with the header's length
                max_length = max(kpi_df[col].astype(str).map(len).max(), len(col))
                # Set the column width to the maximum length plus a little extra for spacing
                rates_sheet.set_column(i, i, max_length + 2)
            kpi_df.to_excel(writer, sheet_name='kpis', index=False)
            print(f"Data successfully exported to {filename}")