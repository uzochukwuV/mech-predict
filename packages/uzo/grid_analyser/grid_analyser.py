import pandas as pd
import numpy as np
import ccxt
import time
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class GridTradingAnalyzer:
    """
    A tool for identifying optimal trading pairs for grid trading strategy,
    recommending grid setup parameters, and projecting possible outcomes.
    """
    
    def __init__(self, exchange_id='binance', api_key=None, api_secret=None):
        """
        Initialize the analyzer with exchange connection and default parameters.
        
        Args:
            exchange_id (str): The exchange to connect to (default: 'binance')
            api_key (str, optional): API key for the exchange
            api_secret (str, optional): API secret for the exchange
        """
        # Initialize exchange connection
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })
        
        # Default parameters for pair selection
        self.selection_params = {
            'min_volume': 1000000,  # Minimum 24h volume in USD
            'max_spread': 0.005,    # Maximum spread (0.5%)
            'min_volatility': 0.02, # Minimum daily volatility (2%)
            'max_volatility': 0.08, # Maximum daily volatility (8%)
            'min_liquidity_depth': 100000,  # Minimum order book depth in USD
            'trend_period': 14,     # Period for trend calculation
            'sideways_threshold': 0.03, # Max range for sideways identification (3%)
        }
        
        # Parameters for grid setup
        self.grid_params = {
            'grid_levels': 10,      # Default number of grid levels
            'risk_factor': 0.1,     # Risk factor for investment sizing (0-1)
            'profit_target_factor': 1.5, # Ratio of potential profit to risk
        }
        
        # Store analyzed pairs
        self.analyzed_pairs = {}
        
    def set_parameters(self, selection_params=None, grid_params=None):
        """Update the parameters used for analysis"""
        if selection_params:
            self.selection_params.update(selection_params)
        if grid_params:
            self.grid_params.update(grid_params)
            
    def get_market_data(self, symbol, timeframe='1d', limit=30):
        """
        Fetch OHLCV data for a specific trading pair
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
            timeframe (str): Timeframe for candles
            limit (int): Number of candles to fetch
            
        Returns:
            pandas.DataFrame: OHLCV data
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_orderbook_depth(self, symbol, limit=20):
        """
        Calculate order book depth
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
            limit (int): Depth of order book to analyze
            
        Returns:
            dict: Bid and ask depths in quote currency
        """
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit)
            
            bid_depth = sum(bid[0] * bid[1] for bid in orderbook['bids'])
            ask_depth = sum(ask[0] * ask[1] for ask in orderbook['asks'])
            
            return {
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'total_depth': bid_depth + ask_depth
            }
        except Exception as e:
            print(f"Error fetching orderbook for {symbol}: {e}")
            return {'bid_depth': 0, 'ask_depth': 0, 'total_depth': 0}
    
    def calculate_volatility(self, df):
        """Calculate daily volatility based on price data"""
        return np.mean(df['high'] / df['low'] - 1)
    
    def identify_trend(self, df):
        """
        Identify market trend using linear regression on closing prices
        
        Returns:
            dict: Trend information including slope, r2, and classification
        """
        # Create feature for regression (x-axis is the day number)
        X = np.array(range(len(df))).reshape(-1, 1)
        y = df['close'].values
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        r_squared = model.score(X, y)
        
        # Normalize slope to percentage change per day
        norm_slope = slope / df['close'].iloc[0]
        
        # Classify trend
        if norm_slope > self.selection_params['sideways_threshold']:
            trend = 'uptrend'
        elif norm_slope < -self.selection_params['sideways_threshold']:
            trend = 'downtrend'
        else:
            trend = 'sideways'
            
        return {
            'slope': norm_slope,
            'r_squared': r_squared,
            'trend': trend
        }
    
    def analyze_pair(self, symbol):
        """
        Complete analysis of a trading pair
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            dict: Analysis results or None if pair doesn't meet criteria
        """
        # Get market data
        df = self.get_market_data(symbol)
        if df is None or len(df) < self.selection_params['trend_period']:
            return None
        
        # Get current ticker
        try:
            ticker = self.exchange.fetch_ticker(symbol)
        except Exception as e:
            print(f"Error fetching ticker for {symbol}: {e}")
            return None
        
        # Calculate spread
        spread = (ticker['ask'] - ticker['bid']) / ticker['bid'] if ticker['bid'] > 0 else float('inf')
        
        # Get order book depth
        depth = self.get_orderbook_depth(symbol)
        
        # Calculate volatility
        volatility = self.calculate_volatility(df)
        
        # Identify trend
        trend_info = self.identify_trend(df)
        
        # Check if pair meets the criteria
        meets_criteria = (
            ticker['quoteVolume'] >= self.selection_params['min_volume'] and
            spread <= self.selection_params['max_spread'] and
            volatility >= self.selection_params['min_volatility'] and
            volatility <= self.selection_params['max_volatility'] and
            depth['total_depth'] >= self.selection_params['min_liquidity_depth']
        )

# advanced        meets_criteria = (
#             ticker['quoteVolume'] >= self.selection_params['min_volume'] and
#             ticker['baseVolume'] >= self.selection_params['min_base_volume'] and
#             average_spread <= self.selection_params['max_spread'] and
#             historical_volatility >= self.selection_params['min_volatility'] and
#             historical_volatility <= self.selection_params['max_volatility'] and
#             depth['total_depth'] >= self.selection_params['min_liquidity_depth'] and
#             abs(current_price - moving_average) / moving_average <= self.selection_params['max_price_deviation'] and
#             not is_strong_trend(ticker['price_history'])
# )
        
        # Store analysis results
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'current_price': ticker['last'],
            'volume_24h': ticker['quoteVolume'],
            'spread': spread,
            'volatility': volatility,
            'liquidity_depth': depth['total_depth'],
            'trend': trend_info['trend'],
            'trend_slope': trend_info['slope'],
            'trend_strength': trend_info['r_squared'],
            'meets_criteria': meets_criteria
        }
        
        # Generate grid recommendations if the pair meets criteria
        if meets_criteria:
            analysis['grid_recommendation'] = self.recommend_grid_setup(
                analysis, df
            )
            analysis['scenario_projections'] = self.project_scenarios(
                analysis, df
            )
            
        self.analyzed_pairs[symbol] = analysis
        return analysis
    
    def recommend_grid_setup(self, analysis, df):
        """
        Generate grid trading setup recommendations
        
        Args:
            analysis (dict): Results from analyze_pair
            df (pandas.DataFrame): Price data
            
        Returns:
            dict: Grid setup recommendations
        """
        current_price = analysis['current_price']
        volatility = analysis['volatility']
        trend = analysis['trend']
        
        # Calculate grid range based on volatility and trend
        if trend == 'uptrend':
            # Skew grid upward for uptrend
            top_margin = volatility * 3.0
            bottom_margin = volatility * 1.5
        elif trend == 'downtrend':
            # Skew grid downward for downtrend
            top_margin = volatility * 1.5
            bottom_margin = volatility * 3.0
        else:  # sideways
            # Symmetric grid for sideways
            top_margin = volatility * 2.0
            bottom_margin = volatility * 2.0
        
        top_price = current_price * (1 + top_margin)
        bottom_price = current_price * (1 - bottom_margin)
        
        # Calculate grid levels and prices
        grid_levels = self.grid_params['grid_levels']
        grid_step = (top_price - bottom_price) / grid_levels
        
        grid_prices = [bottom_price + i * grid_step for i in range(grid_levels + 1)]
        
        # Recommend investment size based on liquidity depth
        liquidity_factor = min(1.0, analysis['liquidity_depth'] / 1000000)  # Scale by liquidity
        base_investment = 10000  # Base investment in USD
        recommended_investment = base_investment * liquidity_factor * self.grid_params['risk_factor']
        
        # Calculate investment per grid level
        investment_per_level = recommended_investment / grid_levels
        
        return {
            'top_price': top_price,
            'bottom_price': bottom_price,
            'grid_levels': grid_levels,
            'grid_step_percentage': grid_step / bottom_price * 100,
            'recommended_investment': recommended_investment,
            'investment_per_level': investment_per_level,
            'grid_prices': grid_prices
        }
    
    def project_scenarios(self, analysis, df):
        """
        Project different market scenarios and their outcomes
        
        Args:
            analysis (dict): Results from analyze_pair
            df (pandas.DataFrame): Price data
            
        Returns:
            dict: Projected outcomes for different scenarios
        """
        grid_rec = analysis['grid_recommendation']
        top_price = grid_rec['top_price']
        bottom_price = grid_rec['bottom_price']
        grid_levels = grid_rec['grid_levels']
        investment = grid_rec['recommended_investment']
        
        # Calculate metrics for each scenario
        scenarios = {}
        
        # Uptrend scenario (price moves from bottom to top)
        uptrend_profits = self._calculate_scenario_profit(
            bottom_price, top_price, grid_levels, investment, 'uptrend'
        )
        
        # Downtrend scenario (price moves from top to bottom)
        downtrend_profits = self._calculate_scenario_profit(
            top_price, bottom_price, grid_levels, investment, 'downtrend'
        )
        
        # Sideways scenario (price oscillates within range)
        sideways_profits = self._calculate_scenario_profit(
            (bottom_price + top_price) / 2, top_price * 0.9, grid_levels / 2, 
            investment / 2, 'sideways'
        )
        
        return {
            'uptrend': {
                'description': f'Price moves from {bottom_price:.2f} to {top_price:.2f}',
                'profit_percentage': uptrend_profits['profit_percentage'],
                'profit_amount': uptrend_profits['profit_amount'],
                'roi': uptrend_profits['roi'],
                'trades_executed': uptrend_profits['trades_executed']
            },
            'downtrend': {
                'description': f'Price moves from {top_price:.2f} to {bottom_price:.2f}',
                'profit_percentage': downtrend_profits['profit_percentage'],
                'profit_amount': downtrend_profits['profit_amount'],
                'roi': downtrend_profits['roi'],
                'trades_executed': downtrend_profits['trades_executed']
            },
            'sideways': {
                'description': f'Price oscillates between {bottom_price:.2f} and {top_price:.2f}',
                'profit_percentage': sideways_profits['profit_percentage'],
                'profit_amount': sideways_profits['profit_amount'],
                'roi': sideways_profits['roi'],
                'trades_executed': sideways_profits['trades_executed']
            }
        }
    
    def _calculate_scenario_profit(self, start_price, end_price, grid_levels, investment, scenario_type):
        """Helper method to calculate profits for different scenarios"""
        grid_step = abs(end_price - start_price) / grid_levels
        investment_per_level = investment / grid_levels
        
        if scenario_type == 'sideways':
            # Assume multiple oscillations for sideways scenario
            oscillations = 3
            trades_executed = int(grid_levels * oscillations * 2)
            avg_profit_per_trade = grid_step / ((start_price + end_price) / 2) * investment_per_level
            total_profit = avg_profit_per_trade * trades_executed
        else:
            # For up/down trends
            trades_executed = int(grid_levels)
            avg_price = (start_price + end_price) / 2
            avg_profit_per_trade = grid_step / avg_price * investment_per_level
            total_profit = avg_profit_per_trade * trades_executed
        
        profit_percentage = total_profit / investment * 100
        roi = total_profit / investment
        
        return {
            'profit_amount': total_profit,
            'profit_percentage': profit_percentage,
            'roi': roi,
            'trades_executed': trades_executed
        }
    
    def scan_all_pairs(self, quote_currency='USDT', limit=50):
        """
        Scan all available pairs with the specified quote currency
        
        Args:
            quote_currency (str): Quote currency to filter pairs (e.g., 'USDT')
            limit (int): Maximum number of pairs to analyze (sorted by volume)
            
        Returns:
            list: Trading pairs that meet the criteria
        """
        try:
            markets = self.exchange.fetch_markets()
            
            pairs = [market['symbol'] for market in markets 
                    if market['quote'] == quote_currency and market['active'] and market["type"]== "spot"]
           
            # Get tickers for all pairs to sort by volume
            tickers = self.exchange.fetch_tickers(pairs)
            
            pair_volumes = [(symbol, tickers[symbol]['quoteVolume'] if tickers[symbol]['quoteVolume'] else 0) 
                           for symbol in pairs]
            
            # Sort by volume and take top pairs
            top_pairs = [pair for pair, volume in sorted(pair_volumes, key=lambda x: x[1], reverse=True)[:limit]]
            
            # Analyze each pair
            suitable_pairs = []
            for symbol in top_pairs:
                print(f"Analyzing {symbol}...")
                result = self.analyze_pair(symbol)
                if result and result['meets_criteria']:
                    suitable_pairs.append(result)
            
            return suitable_pairs
            
        except Exception as e:
            print(f"Error scanning pairs: {e}")
            return []
    def live_monitor(self, pairs, interval=300, duration=3600):
        """
        Continuously monitor the selected pairs and provide live updates
        
        Args:
            pairs (list): List of pairs to monitor
            interval (int): Update interval in seconds
            duration (int): Total monitoring duration in seconds
        
        Returns:
            None
        """
        start_time = time.time()
        iterations = 0
        
        print(f"Starting live monitoring of {len(pairs)} pairs...")
        print(f"Updates will occur every {interval} seconds for {duration/60} minutes")
        
        while time.time() - start_time < duration:
            print(f"\n--- Update #{iterations+1} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
            
            for symbol in pairs:
                previous = self.analyzed_pairs.get(symbol, None)
                current = self.analyze_pair(symbol)
                
                if current is None:
                    continue
                    
                if previous is None:
                    status = "ADDED" if current['meets_criteria'] else "REJECTED"
                else:
                    if current['meets_criteria'] and not previous['meets_criteria']:
                        status = "NEW SIGNAL"
                    elif not current['meets_criteria'] and previous['meets_criteria']:
                        status = "SIGNAL LOST"
                    elif current['meets_criteria']:
                        status = "MAINTAINED"
                    else:
                        status = "STILL UNSUITABLE"
                
                if current['meets_criteria']:
                    grid_rec = current['grid_recommendation']
                    print(f"{symbol} [{status}]: Price=${current['current_price']:.4f} | "
                          f"Trend={current['trend']} | Grid: {grid_rec['bottom_price']:.4f}-{grid_rec['top_price']:.4f} | "
                          f"Invest=${grid_rec['recommended_investment']:.2f}")
                else:
                    print(f"{symbol} [{status}]: No longer meets criteria")
            
            iterations += 1
            
            # Sleep until next update
            next_update = start_time + (iterations * interval)
            sleep_time = max(0, next_update - time.time())
            if sleep_time > 0 and time.time() - start_time < duration:
                print(f"Next update in {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
    
    def visualize_grid(self, symbol):
        """
        Visualize the recommended grid setup for a given pair
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            matplotlib.figure.Figure: Visualization figure
        """
        if symbol not in self.analyzed_pairs or not self.analyzed_pairs[symbol]['meets_criteria']:
            print(f"No valid grid recommendation available for {symbol}")
            return None
        
        analysis = self.analyzed_pairs[symbol]
        grid_rec = analysis['grid_recommendation']
        
        # Get historical data for chart
        df = self.get_market_data(symbol, timeframe='4h', limit=100)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot price history
        ax.plot(df['timestamp'], df['close'], label='Price', color='blue')
        
        # Plot grid levels
        for price in grid_rec['grid_prices']:
            ax.axhline(y=price, linestyle='--', alpha=0.5, color='red')
        
        # Highlight the grid range
        ax.axhspan(grid_rec['bottom_price'], grid_rec['top_price'], alpha=0.2, color='green')
        
        # Add labels and legend
        ax.set_title(f'Grid Trading Analysis for {symbol}', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        
        # Add annotations
        text_info = (
            f"Trend: {analysis['trend']}\n"
            f"Volatility: {analysis['volatility']*100:.2f}%\n"
            f"Grid Range: {grid_rec['bottom_price']:.4f} - {grid_rec['top_price']:.4f}\n"
            f"Grid Levels: {grid_rec['grid_levels']}\n"
            f"Recommended Investment: ${grid_rec['recommended_investment']:.2f}\n"
            f"Expected ROI (sideways): {analysis['scenario_projections']['sideways']['roi']*100:.2f}%"
        )
        
        ax.text(0.02, 0.02, text_info, transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, pairs=None):
        """
        Generate a comprehensive report of analyzed pairs
        
        Args:
            pairs (list, optional): List of specific pairs to include in report
                                   If None, include all analyzed pairs that meet criteria
        
        Returns:
            pandas.DataFrame: Report dataframe
        """
        if pairs is None:
            pairs = [symbol for symbol, data in self.analyzed_pairs.items() 
                    if data['meets_criteria']]
        
        if not pairs:
            print("No suitable pairs found for report")
            return pd.DataFrame()
        
        report_data = []
        
        for symbol in pairs:
            if symbol not in self.analyzed_pairs or not self.analyzed_pairs[symbol]['meets_criteria']:
                continue
                
            analysis = self.analyzed_pairs[symbol]
            grid_rec = analysis['grid_recommendation']
            scenarios = analysis['scenario_projections']
            
            report_data.append({
                'Symbol': symbol,
                'Current Price': analysis['current_price'],
                'Volume (24h)': analysis['volume_24h'],
                'Volatility': f"{analysis['volatility']*100:.2f}%",
                'Trend': analysis['trend'],
                'Grid Bottom': grid_rec['bottom_price'],
                'Grid Top': grid_rec['top_price'],
                'Grid Levels': grid_rec['grid_levels'],
                'Investment': f"${grid_rec['recommended_investment']:.2f}",
                'Per Level': f"${grid_rec['investment_per_level']:.2f}",
                'Uptrend ROI': f"{scenarios['uptrend']['roi']*100:.2f}%",
                'Downtrend ROI': f"{scenarios['downtrend']['roi']*100:.2f}%",
                'Sideways ROI': f"{scenarios['sideways']['roi']*100:.2f}%",
                'Last Updated': analysis['timestamp']
            })
        
        return pd.DataFrame(report_data)

    def save_report(self, filename='grid_trading_report.csv'):
        """Save the current analysis report to a CSV file"""
        report = self.generate_report()
        if not report.empty:
            report.to_csv(filename, index=False)
            print(f"Report saved to {filename}")
        else:
            print("No data available to save")
            
    def export_grid_config(self, symbol, filename=None):
        """
        Export grid configuration for a specific pair
        
        Args:
            symbol (str): Trading pair symbol
            filename (str, optional): Output filename
                                     If None, use symbol_grid_config.json
        
        Returns:
            dict: Grid configuration
        """
        if symbol not in self.analyzed_pairs or not self.analyzed_pairs[symbol]['meets_criteria']:
            print(f"No valid grid recommendation available for {symbol}")
            return None
            
        analysis = self.analyzed_pairs[symbol]
        grid_rec = analysis['grid_recommendation']
        
        # Create config
        config = {
            'symbol': symbol,
            'timestamp': analysis['timestamp'].isoformat(),
            'grid_settings': {
                'upper_price': grid_rec['top_price'],
                'lower_price': grid_rec['bottom_price'],
                'grid_levels': grid_rec['grid_levels'],
                'investment_amount': grid_rec['recommended_investment'],
                'grid_prices': grid_rec['grid_prices']
            },
            'market_conditions': {
                'trend': analysis['trend'],
                'volatility': analysis['volatility'],
                'volume_24h': analysis['volume_24h']
            },
            'projected_outcomes': analysis['scenario_projections']
        }
        
        # Save to file if filename provided
        if filename is None:
            filename = f"{symbol.replace('/', '_')}_grid_config.json"
        
        import json
        with open(filename, 'w') as f:
            json.dump(config, f, indent=4, default=str)
            print(f"Grid configuration exported to {filename}")
            
        return config






def run(args):
    """
    Entry point for agents to use the Grid Trading Analyzer tool.
    
    Args:
        args (dict): Dictionary containing configuration parameters and commands
            Required keys:
                - 'command': Action to perform (scan, analyze, monitor, report)
            Optional keys:
                - 'exchange': Exchange to use (default: 'binance')
                - 'api_key': API key for exchange
                - 'api_secret': API secret for exchange
                - 'quote_currency': Quote currency for pairs (default: 'USDT')
                - 'symbols': List of symbols to analyze
                - 'selection_params': Custom selection parameters
                - 'grid_params': Custom grid parameters
                - 'monitor_interval': Update interval for monitoring in seconds
                - 'monitor_duration': Total monitoring duration in seconds
                - 'output_format': Format for output (json, dataframe, dict)
    
    Returns:
        dict: Results of the operation
    """
    # Validate required arguments
    if 'command' not in args:
        return {'status': 'error', 'message': 'Command is required'}
    
    command = args.get('command').lower()
    
    # Extract configuration parameters
    exchange_id = args.get('exchange', 'binance')
    api_key = args.get('api_key', None)
    api_secret = args.get('api_secret', None)
    quote_currency = args.get('quote_currency', 'USDT')
    symbols = args.get('symbols', [])
    output_format = args.get('output_format', 'dict')
    
    # Initialize analyzer
    try:
        analyzer = GridTradingAnalyzer(
            exchange_id=exchange_id,
            api_key=api_key,
            api_secret=api_secret
        )
        
        # Update parameters if provided
        if 'selection_params' in args:
            analyzer.set_parameters(selection_params=args['selection_params'])
        if 'grid_params' in args:
            analyzer.set_parameters(grid_params=args['grid_params'])
            
        # Execute command
        if command == 'scan':
            # Scan market for suitable pairs
            limit = args.get('limit', 20)
            results = analyzer.scan_all_pairs(quote_currency=quote_currency, limit=limit)
            
            # Process and return results
            output = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'command': 'scan',
                'pairs_analyzed': limit,
                'pairs_found': len(results),
                'results': _format_output(results, output_format)
            }
            
        elif command == 'analyze':
            # Analyze specific pairs
            if not symbols:
                return {'status': 'error', 'message': 'No symbols provided for analysis'}
                
            results = []
            for symbol in symbols:
                analysis = analyzer.analyze_pair(symbol)
                if analysis:
                    results.append(analysis)
            
            # Process and return results
            output = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'command': 'analyze',
                'pairs_analyzed': len(symbols),
                'pairs_suitable': sum(1 for r in results if r['meets_criteria']),
                'results': _format_output(results, output_format)
            }
            
        elif command == 'monitor':
            # Live monitoring of pairs
            if not symbols:
                # If no symbols provided, scan first then monitor
                limit = args.get('limit', 10)
                scan_results = analyzer.scan_all_pairs(quote_currency=quote_currency, limit=limit)
                symbols = [result['symbol'] for result in scan_results if result['meets_criteria']]
            
            if not symbols:
                return {'status': 'error', 'message': 'No suitable pairs found for monitoring'}
                
            # Set up monitoring parameters
            interval = args.get('monitor_interval', 300)  # 5 minutes default
            duration = args.get('monitor_duration', 3600)  # 1 hour default
            
            # Start monitoring in a separate thread or process
            import threading
            
            # Use a dictionary to store results that can be updated by the thread
            monitoring_results = {'updates': [], 'active': True}
            
            def _monitoring_worker():
                start_time = time.time()
                iterations = 0
                
                while time.time() - start_time < duration and monitoring_results['active']:
                    update = {
                        'timestamp': datetime.now().isoformat(),
                        'iteration': iterations + 1,
                        'pairs': {}
                    }
                    
                    for symbol in symbols:
                        previous = analyzer.analyzed_pairs.get(symbol, None)
                        current = analyzer.analyze_pair(symbol)
                        
                        if current is None:
                            continue
                            
                        if previous is None:
                            status = "ADDED" if current['meets_criteria'] else "REJECTED"
                        else:
                            if current['meets_criteria'] and not previous['meets_criteria']:
                                status = "NEW SIGNAL"
                            elif not current['meets_criteria'] and previous['meets_criteria']:
                                status = "SIGNAL LOST"
                            elif current['meets_criteria']:
                                status = "MAINTAINED"
                            else:
                                status = "STILL UNSUITABLE"
                        
                        update['pairs'][symbol] = {
                            'status': status,
                            'meets_criteria': current['meets_criteria'],
                            'price': current['current_price'],
                            'trend': current['trend'],
                            'grid_recommendation': current['grid_recommendation'] if current['meets_criteria'] else None
                        }
                    
                    monitoring_results['updates'].append(update)
                    iterations += 1
                    
                    # Sleep until next update
                    next_update = start_time + (iterations * interval)
                    sleep_time = max(0, next_update - time.time())
                    if sleep_time > 0 and time.time() - start_time < duration:
                        time.sleep(sleep_time)
                
                monitoring_results['completed'] = True
            
            # Start monitoring thread
            monitor_thread = threading.Thread(target=_monitoring_worker)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # Return initial information, agents can call again with 'get_monitoring_results' to get updates
            output = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'command': 'monitor',
                'message': f'Monitoring started for {len(symbols)} pairs',
                'monitor_id': id(monitoring_results),  # Unique ID to reference this monitoring session
                'pairs': symbols,
                'interval': interval,
                'duration': duration,
                'expected_updates': duration // interval
            }
            
            # Store in a global registry for retrieval
            if not hasattr(run, 'monitoring_sessions'):
                run.monitoring_sessions = {}
            run.monitoring_sessions[id(monitoring_results)] = monitoring_results
            
        elif command == 'get_monitoring_results':
            # Retrieve results from an ongoing monitoring session
            monitor_id = args.get('monitor_id')
            if not monitor_id or not hasattr(run, 'monitoring_sessions') or monitor_id not in run.monitoring_sessions:
                return {'status': 'error', 'message': 'Invalid or expired monitoring session ID'}
                
            session = run.monitoring_sessions[monitor_id]
            
            # Process and return results
            output = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'command': 'get_monitoring_results',
                'monitor_id': monitor_id,
                'completed': session.get('completed', False),
                'updates_count': len(session['updates']),
                'updates': session['updates']
            }
            
            # Clean up if completed
            if session.get('completed', False):
                del run.monitoring_sessions[monitor_id]
                
        elif command == 'stop_monitoring':
            # Stop an ongoing monitoring session
            monitor_id = args.get('monitor_id')
            if not monitor_id or not hasattr(run, 'monitoring_sessions') or monitor_id not in run.monitoring_sessions:
                return {'status': 'error', 'message': 'Invalid or expired monitoring session ID'}
                
            run.monitoring_sessions[monitor_id]['active'] = False
            
            output = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'command': 'stop_monitoring',
                'monitor_id': monitor_id,
                'message': 'Monitoring session stopped'
            }
            
        elif command == 'report':
            # Generate a report
            if symbols:
                # Report for specific symbols
                report_data = analyzer.generate_report(symbols)
            else:
                # Report for all analyzed pairs
                report_data = analyzer.generate_report()
            
            # Save report if requested
            if args.get('save_report'):
                filename = args.get('report_filename', 'grid_trading_report.csv')
                report_data.to_csv(filename, index=False)
            
            # Process and return results
            output = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'command': 'report',
                'pairs_count': len(report_data),
                'report': _format_output(report_data, output_format)
            }
            
        elif command == 'export_grid':
            # Export grid configuration
            if not symbols or len(symbols) == 0:
                return {'status': 'error', 'message': 'Symbol required for grid export'}
                
            symbol = symbols[0]  # Take the first symbol
            filename = args.get('export_filename')
            
            config = analyzer.export_grid_config(symbol, filename)
            
            if not config:
                return {'status': 'error', 'message': f'No valid grid configuration for {symbol}'}
                
            # Process and return results
            output = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'command': 'export_grid',
                'symbol': symbol,
                'filename': filename,
                'config': config
            }
            
        elif command == 'visualize':
            # Generate visualization
            if not symbols or len(symbols) == 0:
                return {'status': 'error', 'message': 'Symbol required for visualization'}
                
            symbol = symbols[0]  # Take the first symbol
            fig = analyzer.visualize_grid(symbol)
            
            if not fig:
                return {'status': 'error', 'message': f'No valid grid configuration for {symbol}'}
                
            # Save visualization if requested
            if args.get('save_visualization'):
                filename = args.get('visualization_filename', f'{symbol.replace("/", "_")}_grid.png')
                fig.savefig(filename)
                
            # Process and return results (can't return the figure directly, so return metadata)
            output = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'command': 'visualize',
                'symbol': symbol,
                'message': 'Visualization generated',
                'saved': args.get('save_visualization', False),
                'filename': args.get('visualization_filename') if args.get('save_visualization') else None
            }
            
        else:
            return {'status': 'error', 'message': f'Unknown command: {command}'}
            
        return output
        
    except Exception as e:
        import traceback
        return {
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }

def _format_output(data, format_type):
    """
    Format output data based on the specified format type
    
    Args:
        data: Data to format (list, dict, or DataFrame)
        format_type: Output format (json, dataframe, dict)
        
    Returns:
        Formatted data
    """
    if format_type == 'json':
        import json
        if isinstance(data, pd.DataFrame):
            return json.loads(data.to_json(orient='records', date_format='iso'))
        else:
            return json.loads(json.dumps(data, default=str))
            
    elif format_type == 'dataframe':
        if isinstance(data, pd.DataFrame):
            return data
        else:
            return pd.DataFrame(data)
            
    else:  # dict format (default)
        if isinstance(data, pd.DataFrame):
            return data.to_dict(orient='records')
        else:
            return data
            
# # Example usage:
# if __name__ == "__main__":
#     # Example 1: Scan market for suitable pairs
#     scan_results = run({
#         'command': 'scan',
#         'quote_currency': 'BTC',
#         'limit': 15,
#         'output_format': 'json'
#     })
#     print(f"Found {scan_results['pairs_found']} suitable pairs")
    
#     # Example 2: Analyze specific pairs
#     analyze_results = run({
#         'command': 'analyze',
#         'symbols': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
#         'selection_params': {
#             'min_volatility': 0.01,
#             'max_volatility': 0.09,
#         }
#     })
#     print(f"Analyzed {analyze_results['pairs_analyzed']} pairs, {analyze_results['pairs_suitable']}  are suitable")
    
#     # Example 3: Start monitoring
#     monitor_results = run({
#         'command': 'monitor',
#         'symbols': ['BTC/USDT', 'ETH/USDT'],
#         'monitor_interval': 60,
#         'monitor_duration': 3600
#     })
#     print(f"Monitoring started with ID: {monitor_results['monitor_id']}")
    
#     # Example 4: Generate and save report
#     report_results = run({
#         'command': 'report',
#         'save_report': True,
#         'report_filename': 'current_grid_opportunities.csv'
#     })
#     print(f"Generated report with {report_results['pairs_count']} pairs")

#         # Example: Monitor specific pairs
#     monitor_session = run({
#         'command': 'monitor',
#         'symbols': ['FIL/USDT', 'DOGE/USDT'],
#         'monitor_interval': 30,  # 5 minutes
#         'monitor_duration': 130  # 24 hours
#     })
#     # time.sleep(120)

#     # # Later: Get monitoring results
#     # monitoring_results = run({
#     #     'command': 'get_monitoring_results',
#     #     'monitor_id': monitor_session['monitor_id']
#     # })

#     # # Example: Export grid configuration for a specific pair
#     # grid_config = run({
#     #     'command': 'export_grid',
#     #     'symbols': ['ETH/USDT'],
#     #     'export_filename': 'eth_usdt_grid.json'
#     # })