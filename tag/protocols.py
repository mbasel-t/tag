from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Protocol, Tuple

from gymnasium import spaces


class _Env(Protocol):
    def build(self) -> None: ...

    def reset(self) -> Tuple[Any, Dict[str, Any]]: ...

    def step(self, *actions: Any) -> Tuple[Any, Any, Any, Any, Dict[str, Any]]: ...

    def observe(self) -> Tuple[Any, Dict[str, Any]]: ...


class _Robot(ABC):
    @abstractmethod
    def reset(self) -> None:
        """
        Reset the robot to its initial state
        Does not return obs like env.reset()
        """
        pass

    @abstractmethod
    def act(self):
        """Perform an action on the robot."""
        pass

    @abstractmethod
    def observe(self) -> Dict[str, Any]:
        """Collect observations from the robot."""

    @abstractmethod
    def observation_space(self) -> spaces.Space:
        """Returns the observation space of the robot."""
        pass

    @abstractmethod
    def action_space(self) -> spaces.Space:
        """Returns the action space of the robot."""
        pass


class _Camera(Protocol):
    def render(self, *, names: Iterable[str], **kwargs: Any) -> Dict[str, Any]: ...


class Wraps(ABC):
    """a thing that wraps another thing"""

    @property
    def unwrap(self):
        """recursive unwrap the to base object"""
        if isinstance(self.wrapped, Wraps):
            return self.wrapped.unwrap
        return self.wrapped

    @abstractmethod
    def wrapped(self):
        """returns the thing that is wrapped"""
        pass

    def __getattr__(self, name):
        """if __getattribute__ fails..."""
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.wrapped, name)

    def __dir__(self):
        return list(set(super().__dir__()) | set(dir(self.wrapped)))
