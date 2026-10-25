"""Analytic valuation of a signed fixed-accrual cashflow on an IFR model."""

from typing import cast

import numpy as np

from fixedincomelib.model import ModelType
from fixedincomelib.product import ProductFixedAccruedCashflow
from fixedincomelib.valuation import (
    AnalyticValParam, FundingIndexParameter, ValuationEngineProduct,
    ValuationEngineProductRegistry,
)


class ValuationEngineProductFixedAccrued(ValuationEngineProduct):
    def __init__(self, model, valuation_parameters_collection, product, request):
        super().__init__(model, valuation_parameters_collection, product, request)
        # TODO 3: select the funding curve from the FundingIndexParameter.
        self.product_ = cast(ProductFixedAccruedCashflow, product)

        # The collection exposes has_vp_type() and its store vp_col_ (assumed to be a
        # dict keyed by vp_type); the parameter holds the index in funding_identifier_.
        vp_type = FundingIndexParameter._vp_type
        if not valuation_parameters_collection.has_vp_type(vp_type):
            raise ValueError("A funding index parameter is required for this product.")
        funding_param = cast(FundingIndexParameter, valuation_parameters_collection.vp_col_[vp_type])
        self.funding_index_ = funding_param.funding_identifier_

    @property
    def funding_index(self):
        return self.funding_index_

    def _cash_amount(self) -> float:
        # Signed notional times accrual fraction: C = N * alpha.
        return self.product_.notional * self.product_.accrued

    def _payment_date(self):
        # GUESS: attribute name (the notebook passes payment_date= to the constructor).
        return self.product_.payment_date

    def calculate_value(self) -> None:
        # TODO 4: PV and cash for past, same-day and future payments.
        c = self._cash_amount()
        payment_date = self._payment_date()

        if payment_date > self.value_date:
            self.value_ = c * self.model.discount_factor(self.funding_index, payment_date)
            self.cash_ = 0.0
        elif payment_date == self.value_date:
            self.value_ = c
            self.cash_ = c
        else:
            self.value_ = 0.0
            self.cash_ = 0.0

    def calculate_risk(
        self, gradient: list[np.ndarray], scaler: float = 1.0, accumulate: bool = False,
    ) -> None:
        # TODO 5: grad PV = C * grad D(T_p) for a future payment; zero otherwise.
        # The notebook passes an empty list, so size it here.
        if len(gradient) == 0:
            self.model.resize_gradient(gradient)

        payment_date = self._payment_date()
        if payment_date > self.value_date:
            self.model.discount_factor_gradient_wrt_state(
                self.funding_index,
                payment_date,
                gradient,
                scaler * self._cash_amount(),
                accumulate,
            )
        elif not accumulate:
            # Same-day or past: zero risk. Overwrite mode must clear the blocks;
            # accumulate mode adds nothing.
            for block in gradient:
                block[:] = 0.0


ValuationEngineProductRegistry().register(
    (ModelType.YIELD_CURVE.to_string(), ProductFixedAccruedCashflow._product_type, AnalyticValParam._vp_type),
    ValuationEngineProductFixedAccrued,
)