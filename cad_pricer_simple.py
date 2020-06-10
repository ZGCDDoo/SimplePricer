import datetime as dt
from typing import List, Tuple

import datedelta
import numpy as np
import scipy.optimize as optimize


class SimpleCADPricer:
    def __init__(self):
        pass

    @staticmethod
    def get_ytm_from_price(
            price: float,
            settlement_date: dt.date,
            maturity: dt.date,
            coupon_rate: float,
            frequency: int = 2,
            nominal: float = 100.0,
    ):
        def get_price_root(y):
            return price - SimpleCADPricer.get_clean_price(
                settlement=settlement_date,
                maturity=maturity,
                coupon_rate=coupon_rate,
                discount_rate=y,
                frequency=frequency,
                nominal=nominal,
            )

        ytm_initial_guess: float = 0.01
        return optimize.newton(get_price_root, ytm_initial_guess)

    @staticmethod
    def next_coupon_date(
            settlement: dt.date, maturity: dt.date, frequency: int
    ) -> dt.date:
        assert isinstance(settlement, dt.date)
        assert isinstance(maturity, dt.date)

        next_coupon_dt = SimpleCADPricer.prior_coupon_date(
            settlement, maturity, frequency
        ) + datedelta.datedelta(months=12 // frequency)
        if next_coupon_dt < settlement:
            next_coupon_dt += datedelta.datedelta(months=12 // frequency)
        assert settlement <= next_coupon_dt
        return next_coupon_dt

    @staticmethod
    def prior_coupon_date(
            settlement: dt.date, maturity: dt.date, frequency: int
    ) -> dt.date:
        assert isinstance(settlement, dt.date)
        assert isinstance(maturity, dt.date)

        prior_coupon_dt: dt.date = dt.date(
            dt.datetime.now().year, maturity.month, maturity.day
        )
        if prior_coupon_dt > settlement:
            prior_coupon_dt = prior_coupon_dt - datedelta.datedelta(
                months=12 // frequency
            )

        # Check if we can add a period to the date, to get closest as possible to the settlement date
        while True:
            if (
                    prior_coupon_dt + datedelta.datedelta(months=12 // frequency)
                    < settlement
            ):
                prior_coupon_dt += datedelta.datedelta(months=12 // frequency)
            else:
                break

        assert prior_coupon_dt < settlement

        return prior_coupon_dt

    @staticmethod
    def get_clean_price(
            settlement: dt.date,
            maturity: dt.date,
            coupon_rate: float,
            discount_rate: float,
            frequency: int = 2,
            nominal: float = 100.0,
    ) -> float:

        prior_coupon_dt: dt.date = SimpleCADPricer.prior_coupon_date(
            settlement, maturity, frequency
        )
        next_coupon_dt = SimpleCADPricer.next_coupon_date(settlement, maturity, frequency)
        first_next_coupon = next_coupon_dt

        days_diff = (next_coupon_dt - settlement).days
        w: float = days_diff / (next_coupon_dt - prior_coupon_dt).days

        pv: float = 0.0
        n: int = 0
        while next_coupon_dt != maturity:
            pv_cashflow: float = coupon_rate * nominal / frequency / (
                    1.0 + discount_rate / frequency
            ) ** (n + w)
            pv += pv_cashflow
            n += 1
            next_coupon_dt += datedelta.datedelta(months=12 // frequency)

        pv += (
                nominal
                * (1.0 + coupon_rate / frequency)
                / (1.0 + discount_rate / frequency) ** (n + w)
        )

        return pv - SimpleCADPricer.interest_accrued(
            settlement=settlement,
            maturity=maturity,
            coupon_rate=coupon_rate,
            frequency=frequency,
            nominal=nominal,
        )

    @staticmethod
    def interest_accrued(
            settlement: dt.date,
            maturity: dt.date,
            coupon_rate: float,
            frequency: int = 2,
            nominal: float = 100.0,
    ) -> float:
        if np.isnan(coupon_rate) or np.isnan(frequency):
            return 0.0
        if np.abs(coupon_rate) < 1e-3:
            return 0.0

        prior_coupon_dt: dt.date = SimpleCADPricer.prior_coupon_date(
            settlement, maturity, frequency
        )
        next_coupon_dt = SimpleCADPricer.next_coupon_date(settlement, maturity, frequency)
        accrued_interest: float = coupon_rate * nominal / frequency * (
                settlement - prior_coupon_dt
        ).days / (next_coupon_dt - prior_coupon_dt).days
        return accrued_interest


if __name__ == "__main__":
    settlement: dt.date = dt.date(2020, 5, 12)
    maturity = dt.date(2028, 6, 1)
    coupon_rate = 0.02
    discount_rate = 0.01215
    price = SimpleCADPricer.get_clean_price(settlement, maturity, coupon_rate, discount_rate)
    print(price)
    ytm = SimpleCADPricer.get_ytm_from_price(price, settlement, maturity, coupon_rate)
    print(ytm)
