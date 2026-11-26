
import os
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, DOUBLE_PRECISION
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

# load_dotenv()

# DATABASE_URL = os.getenv(
#     "DATABASE_URL",
#     "postgresql+psycopg2://postgres:postgres@localhost:5432/vehicle_signals",
# )

# engine = create_engine(DATABASE_URL, pool_pre_ping=True)
# SessionLocal = sessionmaker(bind=engine, autoflush=False)
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not configured")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args={"sslmode": "require"},
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
)

class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)  # bcrypt only
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    projects: Mapped[list["Project"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    project_name: Mapped[str] = mapped_column(String(100), nullable=False)
    vehicle_make: Mapped[str] = mapped_column(String(50), nullable=False)
    vehicle_model: Mapped[str] = mapped_column(String(50), nullable=False)
    vehicle_year: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    csv_filename: Mapped[str | None] = mapped_column(String(255))
    row_count: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    user: Mapped["User"] = relationship(back_populates="projects")
    digital_signals: Mapped[list["DigitalSignal"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", passive_deletes=True
    )
    counter_signals: Mapped[list["CounterSignal"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", passive_deletes=True
    )


class DigitalSignal(Base):
    __tablename__ = "digital_signals"
    __table_args__ = (
        UniqueConstraint("project_id", "can_id", "byte_index", "bit_index"),
        CheckConstraint("byte_index BETWEEN 1 AND 8", name="ck_digital_byte"),
        CheckConstraint("bit_index BETWEEN 0 AND 7", name="ck_digital_bit"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    can_id: Mapped[str] = mapped_column(String(20), nullable=False)
    byte_index: Mapped[int] = mapped_column(SmallInteger, nullable=False)  # 1..8 = B1..B8
    bit_index: Mapped[int] = mapped_column(SmallInteger, nullable=False)  # 0 = MSB ... 7 = LSB
    transitions: Mapped[int] = mapped_column(Integer, nullable=False)
    abs_time: Mapped[list[float]] = mapped_column(ARRAY(DOUBLE_PRECISION), nullable=False)
    value: Mapped[list[int]] = mapped_column(ARRAY(SmallInteger), nullable=False)  # 0/1

    project: Mapped["Project"] = relationship(back_populates="digital_signals")


class CounterSignal(Base):
    __tablename__ = "counter_signals"
    __table_args__ = (
        UniqueConstraint("project_id", "can_id", "byte_index"),
        CheckConstraint("byte_index BETWEEN 1 AND 8", name="ck_counter_byte"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    can_id: Mapped[str] = mapped_column(String(20), nullable=False)
    byte_index: Mapped[int] = mapped_column(SmallInteger, nullable=False)  # 1..8 = B1..B8
    min_value: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    max_value: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    step: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    abs_time: Mapped[list[float]] = mapped_column(ARRAY(DOUBLE_PRECISION), nullable=False)
    value: Mapped[list[int]] = mapped_column(ARRAY(SmallInteger), nullable=False)

    project: Mapped["Project"] = relationship(back_populates="counter_signals")


def init_db() -> None:
    """Create the four tables if they do not exist (called once on startup)."""
    Base.metadata.create_all(engine)


def get_db():
    """FastAPI dependency: one session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
