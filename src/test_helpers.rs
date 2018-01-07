macro_rules! assert_vec_eq
{
    ($a: expr, $b: expr) => {
        assert!(($a - $b).mag() < 0.0001, "assertion failed: {:?} ~= {:?}", $a, $b)
    }
}

macro_rules! assert_mat_eq
{
    ($a: expr, $b: expr) => {
        {
            let mut all_elements_equal = true;
            for col in 0..($a.0.len())
            {
                for row in 0..($a.0[0].len())
                {
                    all_elements_equal &= $a.0[col][row] - $b.0[col][row] < 0.0001;
                }
            }
            assert!(all_elements_equal, "assertion failed: {:?} ~= {:?}", $a, $b);
        }
    }
}

macro_rules! assert_quat_eq
{
    ($a: expr, $b: expr) => {
        assert!($a.0 - $b.0 < 0.0001 && ($a.1 - $b.1).mag() < 0.0001, "assertion failed: {:?} ~= {:?}", $a, $b)
    }
}
