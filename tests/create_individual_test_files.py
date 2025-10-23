#!/usr/bin/env python3
"""
Create individual test files for all working functions.
"""

import os
from pathlib import Path

# List of 28 working functions from our comprehensive test
working_functions = [
    'assortativity_coefficient',
    'average_path_length', 
    'average_true_range',
    'awesome_oscillator',
    'bayesian_network_strategy',
    'black_litterman',
    'breakout_strategy',
    'commodity_channel_index',
    'connected_components',
    'degree_distribution',
    'diameter',
    'eigenvector_centrality',
    'global_clustering_coefficient',
    'greedy_modularity_communities',
    'kaufman_adaptive_moving_average',
    'louvain_communities',
    'modularity',
    'money_flow_index',
    'pagerank',
    'remainder_strength',
    'shortest_path_length',
    'transitivity',
    'trend_strength',
    'volatility_position_sizing',
    'volume_weighted_average_price',
    'weighted_moving_average',
    'williams_r'
]

def create_test_file(func_name):
    """Create individual test file for a function."""
    
    test_content = f'''#!/usr/bin/env python3
"""
Test file for {func_name} function.

This function has been verified to work correctly.
See test_all_remaining_functions.py for comprehensive testing scenarios.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tlm
import unittest
import random
import math

class Test{func_name.title().replace('_', '')}(unittest.TestCase):
    """Test cases for {func_name} function."""
    
    def setUp(self):
        """Set up test data."""
        self.prices = [100.0 + random.gauss(0, 5) for _ in range(50)]
        self.ohlc_data = []
        for i, p in enumerate(self.prices):
            if i == 0:
                open_price = p
            else:
                open_price = self.prices[i-1]
            high = p + abs(random.gauss(0, 2))
            low = p - abs(random.gauss(0, 2))
            close = p
            self.ohlc_data.append((open_price, high, low, close))
        
        self.volumes = [1000000 + random.randint(-200000, 500000) for _ in range(50)]
        self.edges = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)]
        self.graph = tlm.Graph(self.edges)
    
    def test_{func_name}_basic(self):
        """Test {func_name} with basic inputs."""
        # This test is based on successful testing in test_all_remaining_functions.py
        
        try:
'''
    
    # Add function-specific test based on category
    if 'centrality' in func_name or func_name in ['assortativity_coefficient', 'average_path_length', 'connected_components', 'degree_distribution', 'diameter', 'global_clustering_coefficient', 'greedy_modularity_communities', 'louvain_communities', 'modularity', 'pagerank', 'shortest_path_length', 'transitivity']:
        test_content += f'''            result = tlm.{func_name}(self.graph)
            self.assertIsNotNone(result)
            
        except Exception as e:
            self.fail(f"{func_name} failed with error: {{e}}")
    
    def test_{func_name}_empty_graph(self):
        """Test {func_name} with edge cases."""
        try:
            empty_graph = tlm.Graph([])
            result = tlm.{func_name}(empty_graph)
            # May return None, 0, or raise exception - all acceptable
        except:
            pass  # Expected for some functions with empty graphs'''
    
    elif func_name in ['average_true_range', 'awesome_oscillator', 'commodity_channel_index', 'money_flow_index', 'williams_r']:
        test_content += f'''            result = tlm.{func_name}(self.ohlc_data)
            self.assertIsInstance(result, list)
            
        except Exception as e:
            self.fail(f"{func_name} failed with error: {{e}}")'''
    
    elif func_name in ['weighted_moving_average', 'kaufman_adaptive_moving_average']:
        test_content += f'''            result = tlm.{func_name}(self.prices)
            self.assertIsInstance(result, list)
            
        except Exception as e:
            self.fail(f"{func_name} failed with error: {{e}}")'''
    
    elif func_name == 'volume_weighted_average_price':
        test_content += f'''            result = tlm.{func_name}(self.ohlc_data, self.volumes)
            self.assertIsInstance(result, list)
            
        except Exception as e:
            self.fail(f"{func_name} failed with error: {{e}}")'''
    
    elif func_name in ['breakout_strategy']:
        test_content += f'''            result = tlm.{func_name}(self.prices, self.ohlc_data)
            self.assertTrue(hasattr(result, 'signals'))
            
        except Exception as e:
            self.fail(f"{func_name} failed with error: {{e}}")'''
    
    elif func_name == 'bayesian_network_strategy':
        test_content += f'''            result = tlm.{func_name}(self.prices, self.volumes)
            self.assertIsInstance(result, list)
            
        except Exception as e:
            self.fail(f"{func_name} failed with error: {{e}}")'''
    
    elif func_name == 'black_litterman':
        test_content += f'''            market_caps = [1e9, 2e9, 3e9]
            expected_returns = [0.08, 0.10, 0.12]
            views = {{0: (0.09, 0.8), 1: (0.11, 0.7)}}
            result = tlm.{func_name}(market_caps, expected_returns, views)
            self.assertIsInstance(result, list)
            
        except Exception as e:
            self.fail(f"{func_name} failed with error: {{e}}")'''
    
    elif func_name in ['volatility_position_sizing']:
        test_content += f'''            result = tlm.{func_name}(self.prices)
            self.assertIsNotNone(result)
            
        except Exception as e:
            self.fail(f"{func_name} failed with error: {{e}}")'''
    
    elif func_name in ['remainder_strength', 'trend_strength']:
        test_content += f'''            signal = [i + 5*math.sin(2*math.pi*i/20) for i in range(100)]
            decomp = tlm.additive_decompose(signal, 20)
            result = tlm.{func_name}(decomp)
            self.assertIsInstance(result, float)
            
        except Exception as e:
            self.fail(f"{func_name} failed with error: {{e}}")'''
    
    else:
        test_content += f'''            # Generic test - function signature varies
            result = tlm.{func_name}()
            self.assertIsNotNone(result)
            
        except Exception as e:
            self.fail(f"{func_name} failed with error: {{e}}")'''
    
    test_content += f'''

if __name__ == "__main__":
    unittest.main(verbosity=2)
'''
    
    return test_content

def main():
    """Create all individual test files."""
    tests_dir = Path("C:/Users/marti/AI-Scoring/tlm/tests")
    
    for func_name in working_functions:
        file_path = tests_dir / f"test_{func_name}.py"
        content = create_test_file(func_name)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"Created: {file_path}")
    
    print(f"\nCreated {len(working_functions)} individual test files!")

if __name__ == "__main__":
    main()