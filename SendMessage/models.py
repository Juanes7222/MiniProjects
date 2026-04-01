from dataclasses import dataclass

@dataclass
class SermonMetadata:
    number: str
    title: str
    preacher: str
    date: str


@dataclass
class SermonLinks:
    youtube: str | None
    spotify: str | None

