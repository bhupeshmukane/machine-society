from .models import Experience


class ExperienceStore:
    def __init__(self) -> None:
        self._experiences: dict[str, Experience] = {}

    def record(self, experience: Experience) -> Experience:
        if experience.experience_id in self._experiences:
            raise ValueError(
                f"Experience already exists: {experience.experience_id}"
            )

        self._experiences[experience.experience_id] = experience
        return experience

    def get(self, experience_id: str) -> Experience | None:
        return self._experiences.get(experience_id)

    def list(self) -> list[Experience]:
        return list(self._experiences.values())

    def count(self) -> int:
        return len(self._experiences)