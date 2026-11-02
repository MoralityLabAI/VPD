from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class SimulatedLedger:
    balances: dict[str, float] = field(default_factory=lambda: {"USDT": 1000.0, "BTC": 0.1})
    positions: dict[str, dict[str, Any]] = field(default_factory=dict)
    transfers: list[dict[str, Any]] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "balances": deepcopy(self.balances),
            "positions": deepcopy(self.positions),
            "transfers": deepcopy(self.transfers),
        }

    def execute(self, tool: str, verb: str, args: Mapping[str, Any]) -> dict[str, Any]:
        if tool == "tradelayer" and verb == "allocate_strategy":
            return self._allocate_strategy(args)
        if tool == "wallet" and verb == "transfer":
            return self._transfer(args)
        if tool == "tradelayer" and verb == "close_strategy":
            return self._close_strategy(args)
        raise ValueError(f"unsupported simulated action: {tool}.{verb}")

    def _allocate_strategy(self, args: Mapping[str, Any]) -> dict[str, Any]:
        asset = str(args.get("asset", "USDT"))
        amount = float(args["amount"])
        strategy_id = str(args["strategy_id"])
        if amount <= 0:
            raise ValueError("amount must be positive")
        if self.balances.get(asset, 0.0) < amount:
            raise ValueError("insufficient balance")
        self.balances[asset] -= amount
        self.positions[strategy_id] = {
            "asset": asset,
            "amount": amount,
            "network": args.get("network", "regtest"),
            "status": "open",
        }
        return {"status": "opened", "strategy_id": strategy_id, "amount": amount}

    def _transfer(self, args: Mapping[str, Any]) -> dict[str, Any]:
        asset = str(args.get("asset", "USDT"))
        amount = float(args["amount"])
        recipient = str(args["recipient"])
        if amount <= 0:
            raise ValueError("amount must be positive")
        if self.balances.get(asset, 0.0) < amount:
            raise ValueError("insufficient balance")
        self.balances[asset] -= amount
        transfer = {"asset": asset, "amount": amount, "recipient": recipient}
        self.transfers.append(transfer)
        return {"status": "sent", **transfer}

    def _close_strategy(self, args: Mapping[str, Any]) -> dict[str, Any]:
        strategy_id = str(args["strategy_id"])
        position = self.positions.get(strategy_id)
        if not position or position.get("status") != "open":
            raise ValueError("strategy is not open")
        position["status"] = "closed"
        self.balances[position["asset"]] = self.balances.get(position["asset"], 0.0) + float(
            position["amount"]
        )
        return {"status": "closed", "strategy_id": strategy_id}
