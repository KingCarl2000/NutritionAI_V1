from pandera import DataFrameSchema, Column, Check, Index, MultiIndex

schema = DataFrameSchema(
    columns={
        "User_ID": Column(
            dtype="int64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=10001159.0, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=19999394.0, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "Gender": Column(dtype="object"),
        "Age": Column(
            dtype="int64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=20.0, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=79.0, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "Height": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=123.0, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=218.0, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "Weight": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=36.0, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=132.0, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "Duration": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=1.0, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=30.0, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "Heart_Rate": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=67.0, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=128.0, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "Body_Temp": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=37.1, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=41.5, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "Calories": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=1.0, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=314.0, raise_warning=False, ignore_na=True
                ),
            ],
        ),
    },
    index=Index(
        dtype="int64",
        checks=None,
        nullable=False,
        coerce=False,
        name=None,
        description=None,
        title=None,
    ),
    coerce=True,
    strict=True,
)
