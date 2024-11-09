from functions import *

data = pd.read_csv("C:\\Users\\isaro\\Downloads\\final.csv")
data['purchase_date'] = pd.to_datetime(data['purchase_date'], errors='coerce')
data['year'] = data['purchase_date'].dt.year
data['month'] = data['purchase_date'].dt.month

#plot_average_monthly_revenue(data)
#plot_average_monthly_sales(data)
#plot_market_share_by_category(data)
#plot_monthly_sales_and_decomposition(data)
#plot_monthly_sales_by_year(data)
#plot_monthly_sales_trends(data)
#plot_revenue_treemap(data)
