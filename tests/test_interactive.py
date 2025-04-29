from glidertest import fetchers, interactive
import pytest

def test_mission_map():
    ds = fetchers.load_sample_dataset()
    interactive.mission_map(ds)

def test_interactive_profile():
    ds = fetchers.load_sample_dataset()
    interactive.interactive_profile(ds)

def test_interactive_up_down_bias():
    ds = fetchers.load_sample_dataset()
    interactive.interactive_up_down_bias(ds)

def test_interactive_daynight_avg():
    ds = fetchers.load_sample_dataset()
    interactive.interactive_daynight_avg(ds)

def test_interactive_ts_plot():
    ds = fetchers.load_sample_dataset()
    interactive.interactive_ts_plot(ds)