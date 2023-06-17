class EpisodeKey:
    """Unlimited buffer"""

    Consumption = 'consumption'
    WorkedHours = 'worked_hours'
    WageRate = 'wage_rate'
    Wealth = 'wealth'
    Ability = 'ability'
    SavingReturn = 'saving_return'
    Asset = 'asset'

    Capital = 'capital'
    Labor = 'labor'

    # Government
    IncomeTax = "income_tax"                     # tau
    IncomeTaxSlope = "income_tax_slope"          # xi
    WealthTax = "wealth_tax"                     # tau_a
    WealthTaxSlope = "income_tax_slope"          # xi_a
    GovernmentSpending = "government_spending"