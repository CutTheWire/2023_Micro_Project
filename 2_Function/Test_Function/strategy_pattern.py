from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

class Context():
    """
    컨텍스트(Context)는 클라이언트에게 관심 있는 인터페이스를 정의합니다.
    """

    def __init__(self, strategy: Strategy) -> None:
        """
        일반적으로 컨텍스트는 생성자를 통해 전략을 받지만,
        런타임에 전략을 변경할 수 있도록 세터(setter)를 제공하기도 합니다.
        """

        self._strategy = strategy

    @property
    def strategy(self) -> Strategy:
        """
        컨텍스트는 Strategy 객체 중 하나를 참조합니다. 컨텍스트는 전략의 구체적인 클래스를 알지 않습니다.
        전략 인터페이스를 통해 모든 전략과 작업을 수행합니다.
        """

        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
        """
        일반적으로 컨텍스트는 런타임에 Strategy 객체를 교체할 수 있도록 합니다.
        """

        self._strategy = strategy

    def do_some_business_logic(self) -> None:
        """
        컨텍스트는 여러 버전의 알고리즘을 직접 구현하는 대신, 전략 객체에 일부 작업을 위임합니다.
        """

        # ...

        print("컨텍스트: 전략을 사용하여 데이터 정렬 (어떻게 동작할지는 모름)")
        result = self._strategy.do_algorithm(["a", "b", "c", "d", "e"])
        print(",".join(result))

        # ...

class Strategy(ABC):
    """
    Strategy 인터페이스는 일부 알고리즘의 모든 지원 버전에 공통된 작업을 선언합니다.
    컨텍스트는 Concrete Strategies에서 정의된 알고리즘을 호출하는 데 이 인터페이스를 사용합니다.
    """

    @abstractmethod
    def do_algorithm(self, data: List):
        pass

"""
Concrete Strategies는 기본 Strategy 인터페이스를 따르면서 알고리즘을 구현합니다.
인터페이스를 통해 컨텍스트에서 서로 교환할 수 있도록 합니다.
"""

class ConcreteStrategyA(Strategy):
    def do_algorithm(self, data: List) -> List:
        return sorted(data)

class ConcreteStrategyB(Strategy):
    def do_algorithm(self, data: List) -> List:
        return reversed(sorted(data))

if __name__ == "__main__":
    # 클라이언트 코드는 구체적인 전략을 선택하고 컨텍스트에 전달합니다.
    # 클라이언트는 전략 간의 차이를 알아야 올바른 선택을 할 수 있습니다.

    context = Context(ConcreteStrategyA())
    print("클라이언트: 일반 정렬 전략을 선택했습니다.")
    context.do_some_business_logic()
    print()

    print("클라이언트: 역순 정렬 전략을 선택했습니다.")
    context.strategy = ConcreteStrategyB()
    context.do_some_business_logic()
