use vecx::*;

#[test]
fn it_removes_value_from_one_vector_and_inserts_it_into_another() {
    let mut vec1 = type_erased_vec![i32, 1, 2, 3];
    let value = vec1.remove(1);
    assert_eq!(2, vec1.len());

    let mut vec2 = type_erased_vec![i32, 1, 3];
    unsafe { vec2.insert(1, value) };
    assert_eq!(3, vec2.len());
}
