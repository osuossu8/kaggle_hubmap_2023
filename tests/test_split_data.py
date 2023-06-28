import os
import sys

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

sys.path.append(os.getcwd())
from src.split_data import BaseConfig, DataSplitter


class TestDataSplitter:
    def setup_class(self):
        self.CFG = BaseConfig(42, 5, "group", "y")
        self.MultiTargetCFG = BaseConfig(42, 5, "group", ["y1", "y2", "y3"])

    def test_split_kfold(self):
        # Arrange
        input_df = pd.DataFrame()
        input_df["y"] = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]
        input_df["x"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

        expect_df = pd.DataFrame()
        expect_df["y"] = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]
        expect_df["x"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        expect_df["kfold"] = [0, 2, 2, 4, 3, 1, 4, 3, 1, 0, 3, 0, 4, 1, 2]

        # Act
        actual_df = DataSplitter.split_kfold(input_df, self.CFG)

        # Assert
        assert_frame_equal(expect_df, actual_df)

    def test_split_stratified(self):
        # Arrange
        input_df = pd.DataFrame()
        input_df["y"] = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]
        input_df["x"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

        expect_df = pd.DataFrame()
        expect_df["y"] = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]
        expect_df["x"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        expect_df["kfold"] = [0, 2, 4, 0, 0, 4, 3, 3, 1, 2, 2, 1, 1, 3, 4]

        # Act
        actual_df = DataSplitter.split_stratified(input_df, self.CFG)

        # Assert
        assert_frame_equal(expect_df, actual_df)

    def test_split_group(self):
        # Arrange
        input_df = pd.DataFrame()
        input_df["group"] = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
        input_df["y"] = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]
        input_df["x"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

        expect_df = pd.DataFrame()
        expect_df["group"] = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
        expect_df["y"] = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]
        expect_df["x"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        expect_df["kfold"] = [4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0]

        # Act
        actual_df = DataSplitter.split_group(input_df, self.CFG)

        # Assert
        assert_frame_equal(expect_df, actual_df)

    def test_split_stratified_group(self):
        # Arrange
        input_df = pd.DataFrame()
        input_df["group"] = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
        input_df["y"] = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]
        input_df["x"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

        expect_df = pd.DataFrame()
        expect_df["group"] = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
        expect_df["y"] = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]
        expect_df["x"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        expect_df["kfold"] = [3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 4, 4, 4]

        # Act
        actual_df = DataSplitter.split_stratified_group(input_df, self.CFG)

        # Assert
        assert_frame_equal(expect_df, actual_df)

    def test_split_multilabel_stratified(self):
        # Arrange
        input_df = pd.DataFrame()
        input_df["y1"] = [0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1]
        input_df["y2"] = [0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0]
        input_df["y3"] = [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1]
        input_df["x"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

        expect_df = pd.DataFrame()
        expect_df["y1"] = [0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1]
        expect_df["y2"] = [0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0]
        expect_df["y3"] = [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1]
        expect_df["x"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        expect_df["kfold"] = [1, 1, 2, 1, 4, 0, 0, 3, 4, 4, 0, 2, 3, 2, 3]

        # Act
        actual_df = DataSplitter.split_multilabel_stratified(
            input_df, self.MultiTargetCFG
        )

        # Assert
        assert_frame_equal(expect_df, actual_df)
