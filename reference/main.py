from settings import settings
from object_factory import object_factory
from mappers import portfolios_allocation_mapper
def generate_optimum_portfolio():
#instantiate the objects with the settings
    obj_factory = object_factory(settings)
    ce = obj_factory.get_companies_extractor()
    cp = obj_factory.get_charts_plotter()
    mcs = obj_factory.get_portfolio_generator()
    fr = obj_factory.get_file_repository()
    mc = obj_factory.get_metrics_calculator()
#1. Get companies
    companies = ce.get_companies_list()
#2. Get company stock prices
    
    end_date = settings.get_end_date()
    start_date = settings.get_start_date(end_date)
    closing_prices = obj_factory.get_price_extractor(companies).get_prices(settings.PriceEvent, start_date, end_date)
#plot stock prices & save data to a file
    cp.plot_prices(closing_prices)    
    fr.save_to_file(closing_prices, 'StockPrices')
#3. Calculate Daily Returns    
    returns = settings.DailyAssetsReturnsFunction(closing_prices, settings.ReturnType)
    #plot stock prices & save data to a file
    cp.plot_returns(returns)
    fr.save_to_file(returns, 'Returns')
#4. Calculate Expected Mean Return & Covariance
    expected_returns = settings.AssetsExpectedReturnsFunction(returns)
    covariance = settings.AssetsCovarianceFunction(returns)
    #Plot & Save covariance to file
    cp.plot_correlation_matrix(returns)
    fr.save_to_file(covariance, 'Covariances')
#5. Use Monte Carlo Simulation
    #Generate portfolios with allocations
    portfolios_allocations_df = mcs.generate_portfolios(expected_returns, covariance, settings.RiskFreeRate)
    portfolio_risk_return_ratio_df = portfolios_allocation_mapper.map_to_risk_return_ratios(portfolios_allocations_df)
    
    #Plot portfolios, print max sharpe portfolio & save data
    cp.plot_portfolios(portfolio_risk_return_ratio_df)
    max_sharpe_portfolio = mc.get_max_sharpe_ratio(portfolio_risk_return_ratio_df)['Portfolio']
    max_shape_ratio_allocations = portfolios_allocations_df[[ 'Symbol', max_sharpe_portfolio]]
    print(max_shape_ratio_allocations)
    fr.save_to_file(portfolios_allocations_df, 'MonteCarloPortfolios')
    fr.save_to_file(portfolio_risk_return_ratio_df, 'MonteCarloPortfolioRatios')   
    
    fr.close()


#This function will execute the steps above
generate_optimum_portfolio()