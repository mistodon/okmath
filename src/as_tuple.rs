pub trait AsTuple
{
    type Tuple;

    fn as_tuple(&self) -> Self::Tuple;
}

impl<T: Copy> AsTuple for [T; 2]
{
    type Tuple = (T, T);

    fn as_tuple(&self) -> Self::Tuple { (self[0], self[1]) }
}

impl<T: Copy> AsTuple for [T; 3]
{
    type Tuple = (T, T, T);

    fn as_tuple(&self) -> Self::Tuple { (self[0], self[1], self[2]) }
}

impl<T: Copy> AsTuple for [T; 4]
{
    type Tuple = (T, T, T, T);

    fn as_tuple(&self) -> Self::Tuple { (self[0], self[1], self[2], self[3]) }
}


#[cfg(test)]
mod tests
{
    use super::*;

    #[test]
    fn as_tuple_for_arrays()
    {
        let a2 = [1, 2];
        let a3 = [1, 2, 3];
        let a4 = [1, 2, 3, 4];

        assert_eq!(a2.as_tuple(), (1, 2));
        assert_eq!(a3.as_tuple(), (1, 2, 3));
        assert_eq!(a4.as_tuple(), (1, 2, 3, 4));
    }
}

