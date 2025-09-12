from ppo_on_grid2op.baseline import train_baseline

TEST_ENV = "l2rpn_case14_sandbox"


def test_train_baseline():
    train_baseline(TEST_ENV, iterations=100)
