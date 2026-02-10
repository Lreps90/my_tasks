
def lifetime_isa_extended(initial_deposit=0, monthly_contribution=100, saving_years=14, growth_years=10, aer=0.025):
    months_saving = saving_years * 12 + 9
    months_growth = growth_years * 12
    monthly_interest = (1 + aer) ** (1 / 12) - 1  # Convert AER to monthly rate

    total_contributed = initial_deposit
    total_bonus = 0
    total_balance = initial_deposit

    # Contribution phase (14 years)
    for month in range(1, months_saving + 1):
        total_contributed += monthly_contribution

        # Apply 25% government bonus (monthly, capped at £4000/year)
        monthly_bonus = min(monthly_contribution, 4000 / 12) * 0.25
        total_bonus += monthly_bonus
        total_balance += monthly_contribution + monthly_bonus

        # Apply monthly interest
        total_balance *= (1 + monthly_interest)

    print(f"After {saving_years} years of contributions:")
    print(f" - Total Invested (Excluding Bonus): £{total_contributed:.2f}")
    print(f" - Total Government Bonus: £{total_bonus:.2f}")
    print(f" - Balance before growth period: £{total_balance:.2f}\n")

    # Growth phase (10 years, no further contributions or bonuses)
    for _ in range(months_growth):
        total_balance *= (1 + monthly_interest)

    total_money_made = total_balance - total_contributed  # Includes bonus + interest

    print(f"After an additional {growth_years} years of growth (no contributions):")
    print(f" - Final Amount Available: £{total_balance:.2f}")
    print(f" - Total Money Made (Bonus + Interest): £{total_money_made:.2f}")

# Run the function
#lifetime_isa_extended()

import matplotlib.pyplot as plt

def lifetime_isa_dynamic(
    initial_deposit=1000,
    monthly_contribution=100,
    saving_years=14,
    growth_years=10,
    aer_first_year=4.7,  # in %
    aer_after=2.5  # in %
):
    # Convert AER to monthly interest rates
    months_saving = saving_years * 12
    months_growth = growth_years * 12
    monthly_interest_first_year = (1 + aer_first_year / 100) ** (1/12) - 1
    monthly_interest_after = (1 + aer_after / 100) ** (1/12) - 1

    total_contributed = initial_deposit
    total_bonus = 0
    total_balance = initial_deposit
    balance_over_time = []  # To track balance for graphing

    # Contribution phase (saving period)
    for month in range(1, months_saving + 1):
        total_contributed += monthly_contribution

        # Apply 25% government bonus (monthly, up to £4,000 per year)
        monthly_bonus = min(monthly_contribution, 4000 / 12) * 0.25
        total_bonus += monthly_bonus
        total_balance += monthly_contribution + monthly_bonus

        # Apply interest (First year at 4.7%, then 2.5%)
        if month <= 12:
            total_balance *= (1 + monthly_interest_first_year)
        else:
            total_balance *= (1 + monthly_interest_after)

        balance_over_time.append(total_balance)

    print(f"\nAfter {saving_years} years of contributions:")
    print(f" - Total Invested (Excluding Bonus): £{total_contributed:.2f}")
    print(f" - Total Government Bonus: £{total_bonus:.2f}")
    print(f" - Balance before growth period: £{total_balance:.2f}\n")

    # Growth phase (investment continues with no contributions)
    for _ in range(months_growth):
        total_balance *= (1 + monthly_interest_after)
        balance_over_time.append(total_balance)

    total_money_made = total_balance - total_contributed  # Includes bonus + interest

    print(f"After an additional {growth_years} years of growth (no contributions):")
    print(f" - Final Amount Available: £{total_balance:.2f}")
    print(f" - Total Money Made (Bonus + Interest): £{total_money_made:.2f}")

    # Plot growth over time
    plt.figure(figsize=(10, 5))
    plt.plot(balance_over_time, label="Balance Over Time", color="blue")
    plt.xlabel("Months")
    plt.ylabel("Total Balance (£)")
    plt.title("Lifetime ISA Growth Over Time")
    plt.legend()
    plt.grid(True)
    #plt.show()

# Run the function with your parameters
lifetime_isa_dynamic(
    initial_deposit=0,
    monthly_contribution=300, # Max £4000 per yer for 25% gov top-up (max top up of £1000).
    saving_years=14,
    growth_years=10,
    aer_first_year=2.5,  # First-year interest rate
    aer_after=2.5  # Interest rate after first year
)
