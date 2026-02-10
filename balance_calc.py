# Initial setup
initial_balance = 3500    # Initial balance in the account
monthly_deposit = 200    # Amount deposited on the 1st of each month
annual_interest_rate = 0.08  # 4% AER/GROSS
daily_interest_rate = annual_interest_rate / 365  # Daily interest rate
months = 12*19      # Duration in months

import math

# Calculate balance without interest: initial balance plus deposits only
balance_without_interest = initial_balance + monthly_deposit * months

# Calculate balance with interest using daily compounding
balance_with_interest = initial_balance

for month in range(months):
    # Add the monthly deposit on the 1st of the month
    balance_with_interest += monthly_deposit

    # Calculate daily interest for the month
    # Approximating days per month: 30 days for April, June, September, November; 31 otherwise
    days_in_month = 30 if month in [3, 5, 8, 10] else 31
    for day in range(days_in_month):
        balance_with_interest += balance_with_interest * daily_interest_rate

# Round the results for clarity
final_balance = round(balance_with_interest, 2)
non_interest_balance = round(balance_without_interest, 2)
interest_earned = round(final_balance - non_interest_balance, 2)

# Output the results
print(f"Final balance with interest: £{final_balance}")
print(f"Balance without interest: £{non_interest_balance}")
print(f"Total interest earned: £{interest_earned}")
