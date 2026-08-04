from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Position:
    symbol: str
    quantity: float
    average_entry_price: float


@dataclass(frozen=True)
class PortfolioSnapshot:
    starting_capital: float
    cash: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    positions: dict[str, Position]


@dataclass
class Portfolio:
    starting_capital: float
    cash: float = field(init=False)
    realized_pnl: float = field(default=0.0, init=False)
    positions: dict[str, Position] = field(
        default_factory=dict,
        init=False,
    )

    def __post_init__(self) -> None:
        if self.starting_capital <= 0:
            raise ValueError(
                "Starting capital must be greater than zero."
            )

        self.cash = self.starting_capital

    def buy(
        self,
        *,
        symbol: str,
        quantity: float,
        price: float,
    ) -> Position:
        self._validate_order(
            symbol=symbol,
            quantity=quantity,
            price=price,
        )

        cost = quantity * price

        if cost > self.cash:
            raise ValueError(
                "Insufficient cash for purchase."
            )

        existing = self.positions.get(symbol)

        if existing is None:
            position = Position(
                symbol=symbol,
                quantity=quantity,
                average_entry_price=price,
            )
        else:
            total_quantity = (
                existing.quantity + quantity
            )

            total_cost = (
                existing.quantity
                * existing.average_entry_price
                + cost
            )

            position = Position(
                symbol=symbol,
                quantity=total_quantity,
                average_entry_price=(
                    total_cost / total_quantity
                ),
            )

        self.positions[symbol] = position
        self.cash -= cost

        return position

    def sell(
        self,
        *,
        symbol: str,
        quantity: float,
        price: float,
    ) -> float:
        self._validate_order(
            symbol=symbol,
            quantity=quantity,
            price=price,
        )

        position = self.positions.get(symbol)

        if position is None:
            raise ValueError(
                f"No open position for '{symbol}'."
            )

        if quantity > position.quantity:
            raise ValueError(
                "Cannot sell more than the "
                "current position."
            )

        proceeds = quantity * price

        realized = (
            price - position.average_entry_price
        ) * quantity

        self.cash += proceeds
        self.realized_pnl += realized

        remaining_quantity = (
            position.quantity - quantity
        )

        if remaining_quantity == 0:
            del self.positions[symbol]
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=remaining_quantity,
                average_entry_price=(
                    position.average_entry_price
                ),
            )

        return realized

    def unrealized_pnl(
        self,
        market_prices: dict[str, float],
    ) -> float:
        total = 0.0

        for symbol, position in self.positions.items():
            if symbol not in market_prices:
                raise ValueError(
                    f"Missing market price for '{symbol}'."
                )

            total += (
                market_prices[symbol]
                - position.average_entry_price
            ) * position.quantity

        return total

    def equity(
        self,
        market_prices: dict[str, float],
    ) -> float:
        return (
            self.cash
            + sum(
                position.quantity
                * market_prices[symbol]
                for symbol, position
                in self.positions.items()
                if symbol in market_prices
            )
        )

    def snapshot(
        self,
        market_prices: dict[str, float],
    ) -> PortfolioSnapshot:
        unrealized = self.unrealized_pnl(
            market_prices
        )

        return PortfolioSnapshot(
            starting_capital=self.starting_capital,
            cash=self.cash,
            equity=self.equity(market_prices),
            realized_pnl=self.realized_pnl,
            unrealized_pnl=unrealized,
            positions={
                symbol: Position(
                    symbol=position.symbol,
                    quantity=position.quantity,
                    average_entry_price=(
                        position.average_entry_price
                    ),
                )
                for symbol, position
                in self.positions.items()
            },
        )

    @staticmethod
    def _validate_order(
        *,
        symbol: str,
        quantity: float,
        price: float,
    ) -> None:
        if not symbol.strip():
            raise ValueError(
                "Symbol cannot be empty."
            )

        if quantity <= 0:
            raise ValueError(
                "Quantity must be greater than zero."
            )

        if price <= 0:
            raise ValueError(
                "Price must be greater than zero."
            )