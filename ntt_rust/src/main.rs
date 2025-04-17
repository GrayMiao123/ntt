use rand::Rng;
use rayon::prelude::*;
use std::io::{self, Write};
use std::time::Instant;
use rand::SeedableRng;

const P: u64 = 998244353;// A famous bigg prime 2^23 * 119 + 1
const G: u64 = 3;   // The generator 

// Fast exponentiation algorithm https://homepages.math.uic.edu/~leon/cs-mcs401-s08/handouts/fastexp.pdf
fn qpow(mut base: u64, mut power: u64) -> u64 {
    let mut result: u64 = 1;
    while power > 0 {
        if power & 1 != 0 {
            result = (result * base) % P;
        }
        base = (base * base) % P;
        power >>= 1;
    }
    result
}

//Find least 2^k such that 2^k > n 
fn expand(n: usize) -> usize {
    let mut m = 1;
    while m < n {
        m <<= 1;
    }
    m
}

//Parallel NTT op=1 means forward -1 means inverse
fn ntt(a: &mut [u64], op: i32) {
    let n = a.len();

    //bit reverse
    let mut r = vec![0usize; n];
    for i in 0..n {
        r[i] = (r[i >> 1] >> 1) | if i & 1 != 0 { n >> 1 } else { 0 };
    }
    for i in 0..n {
        if i < r[i] {
            a.swap(i, r[i]);
        }
    }

    let g = G;
    let gi = qpow(g, P - 2);//Compute the inverse of g which is g^(-1) by fermat little theorem
    //Assign the multiplication factor befor butterflies algorithm
    let mut g1_values = Vec::new();
    let mut size = 2;
    while size <= n {
        let exp = (P - 1) / (size as u64);
        let base = if op == 1 { g } else { gi };
        g1_values.push(qpow(base, exp));
        size <<= 1;
    }

    let inv_n = if op == -1 { qpow(n as u64, P - 2) } else { 1 };

    // Transfer the pointer to a as usize，since integer can be used among threads under unsafe scope
    let a_ptr_usize = a.as_mut_ptr() as usize;

    // Butterflies algorithm from bottom to up
    let mut g1_idx = 0;
    let mut i = 2;
    while i <= n {
        let g1 = g1_values[g1_idx];//obtain the mutiplication factor
        // Use rayon to do parallel ntt 
        (0..(n / i)).into_par_iter().for_each(|j| {
            let start = j*i;//obtain the start position offset for the pointer
            let mut gk = 1u64;
            unsafe {
                // Transfer integer a_ptr_usize back to raw pointer
                let ptr = (a_ptr_usize + start * std::mem::size_of::<u64>()) as *mut u64;
                for k in 0..(i / 2){
                    let x = *ptr.add(k);
                    let y = (gk * *ptr.add(k +i/2)) % P;
                    *ptr.add(k) = (x + y) % P;
                    *ptr.add(k + i / 2) = (x+P-y) % P;
                    gk = (gk * g1) % P;
                }
            }
        });
        g1_idx += 1;
        i <<= 1;
    }

    if op == -1 {
        a.par_iter_mut().for_each(|x| {
            *x = (*x * inv_n) % P;
        });
    }
}


fn serial_ntt(a: &mut [u64], op: i32) {
    let n = a.len();
    let mut r = vec![0usize; n];
    for i in 0..n {
        r[i] = (r[i >> 1] >> 1) | if i & 1 != 0 { n >> 1 } else { 0 };
    }
    for i in 0..n {
        if i < r[i] {
            a.swap(i, r[i]);
        }
    }
    let g = G;
    let gi = qpow(g, P - 2);
    let mut size = 2;
    while size <= n {
        let exp = (P - 1) / (size as u64);
        let base = if op == 1 { g } else { gi };
        let g1 = qpow(base, exp);
        for j in (0..n).step_by(size) {
            let mut gk = 1u64;
            for k in 0..(size / 2) {
                let x = a[j + k];
                let y = (gk * a[j + k + size / 2]) % P;
                a[j + k] = (x + y) % P;
                a[j + k + size / 2] = (x + P - y) % P;
                gk = (gk * g1) % P;
            }
        }
        size <<= 1;
    }
    if op == -1 {
        let inv_n = qpow(n as u64, P - 2);
        for x in a.iter_mut() {
            *x = (*x*inv_n) % P;
        }
    }
}

//Parallel Polynomial multiplication
fn polynomial_multiply(a: &[u64], b: &[u64]) -> Vec<u64> {
    let na = a.len();
    let nb = b.len();
    let n = na + nb - 1;
    let m = expand(n);

    let mut fa = a.to_vec();
    let mut fb = b.to_vec();
    fa.resize(m, 0);
    fb.resize(m, 0);

    //get the NTT transform form
    ntt(&mut fa, 1);
    ntt(&mut fb, 1);

     //do the element wise multiplication
    fa.par_iter_mut()
        .zip(fb.par_iter())
        .for_each(|(x, &y)| *x = (*x * y) % P);

    //inverse ntt transform back to original form
    ntt(&mut fa, -1);
    fa.resize(n, 0);
    fa
}


fn serial_polynomial_multiply(a: &[u64], b: &[u64]) -> Vec<u64> {
    let na = a.len();
    let nb = b.len();
    let n = na + nb - 1;
    let m = expand(n);

    let mut fa = a.to_vec();
    let mut fb = b.to_vec();
    fa.resize(m, 0);
    fb.resize(m, 0);

    serial_ntt(&mut fa, 1);
    serial_ntt(&mut fb, 1);

    for i in 0..m {
        fa[i] = (fa[i] * fb[i]) % P;
    }
    serial_ntt(&mut fa, -1);
    fa.resize(n, 0);
    fa
}


fn print_polynomial(poly: &[u64]) {
    for (i, coef) in poly.iter().enumerate() {
        if i == 0 {
            print!("{}", coef);
        } else {
            print!(" + {}x^{}", coef, i);
        }
    }
    println!();
}


fn read_polynomial() -> Vec<u64> {
    println!("Input the highest order of polynomial：(3 means order3)");
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let degree: usize = input.trim().parse().expect("Please input an integer！");
    println!("Write down the coefficient from lowest order to highest order, between each coefficient we need space");
    input.clear();
    io::stdin().read_line(&mut input).unwrap();
    input
        .trim()
        .split_whitespace()
        .take(degree + 1)
        .map(|s| s.parse::<u64>().expect("Invalid input"))
        .collect()
}
// Generate random polynomial 
fn generate_random_polynomial(degree: usize, seed: u64) -> Vec<u64> {
    let mut poly = vec![0u64; degree + 1];
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    for coef in poly.iter_mut() {
        *coef = rng.gen_range(0..1000);
    }
    poly
}

// Compare if two polynomial are equal?
fn polynomials_equal(a: &[u64], b: &[u64]) -> bool {
    a.len() == b.len() && b.iter().zip(b.iter()).all(|(&c, &d)| c == d)
}

fn main() {
    println!("Polynimial Multiplication program");
    println!("----------------------------------------");
    println!("Choose what you wan to do?");
    println!("1. Input two polynomial by hand");
    println!("2. Accuracy test（Serial vs Parrallel）");
    println!("3. Runtime test with large scale polynomial");
    io::stdout().flush().unwrap();

    let mut choice_str = String::new();
    io::stdin().read_line(&mut choice_str).unwrap();
    let choice: u32 = choice_str.trim().parse().unwrap_or(0);

    match choice {
        1 => {
            println!("Input first polynomial：");
            let poly_a = read_polynomial();
            println!("Input second polynomial：");
            let poly_b = read_polynomial();
            println!("How many threads you need?：(0 means serial)");
            let mut thread_str = String::new();
            io::stdin().read_line(&mut thread_str).unwrap();
            let num_threads: usize = thread_str.trim().parse().unwrap_or(0);

            let result = if num_threads == 0 {
                serial_polynomial_multiply(&poly_a, &poly_b)
            } else {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .build()
                    .unwrap();
                pool.install(|| polynomial_multiply(&poly_a, &poly_b))
            };

            println!("Misssion complete！");
            println!("----------------------------------------");
            println!("The polynomial after multiplication is：");
            print_polynomial(&result);
        }
        2 => {
            println!("Accuracy test");
            println!("----------------------------------------");
            let test_sizes = [1 <<8, 1 << 16, 1 << 20, 1 << 24, 1 << 26];
            println!("We will test with following sclae：");
            for size in &test_sizes {
                print!("{} ", size);
            }
            println!();
            let thread_counts = [1, 2, 4, 8, 16];
            println!("Thread Number：");
            for &t in &thread_counts {
                print!("{} ", t);
            }
            println!();
            println!("----------------------------------------");

            let seed = Instant::now().elapsed().as_secs(); 
            println!("Set random seed:: {}", seed);

            for &size in &test_sizes {
                println!("Test scale size:  {} order polynomial", size);
                let poly_a = generate_random_polynomial(size, seed);
                let poly_b = generate_random_polynomial(size, seed + 1);
                println!("Serial output...");
                let serial_result = serial_polynomial_multiply(&poly_a, &poly_b);

                for &threads in &thread_counts {
                    println!("  Use {} threads...", threads);
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .unwrap();
                    let parallel_result =
                        pool.install(|| polynomial_multiply(&poly_a, &poly_b));
                    if polynomials_equal(&serial_result, &parallel_result) {
                        println!("   Result is consistent with serial version.");
                    } else {
                        println!("  our opnemp version is not correct");
                    }
                }
                println!("----------------------------------------");
            }
        }
        3 => {
            println!("Run time test");
            println!("----------------------------------------");
            println!("Input how many threads?(0 means serial version)：");
            let mut thread_str = String::new();
            io::stdin().read_line(&mut thread_str).unwrap();
            let num_threads: usize = thread_str.trim().parse().unwrap_or(0);
            let test_sizes = [1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 26];
            println!("We will test following different order scale：");
            for size in &test_sizes {
                print!("{} ", size);
            }
            println!();

            for &size in &test_sizes {
                println!("Test size: {} order polynomial", size);
                let mut rng = rand::thread_rng();
                let poly_a: Vec<u64> = (0..size).map(|_| rng.gen_range(0..1000)).collect();
                let poly_b: Vec<u64> = (0..size).map(|_| rng.gen_range(0..1000)).collect();

                if num_threads == 0 {
                    println!("Start serial computation...");
                    let start = Instant::now();
                    let _res = serial_polynomial_multiply(&poly_a, &poly_b);
                    let duration = start.elapsed();
                    let ms = duration.as_secs_f64() * 1000.0;
                    println!(
                        "Serial Mission comoplete！execution time：{:.2} milliseconds{}",
                        ms,
                        if ms > 1000.0 {
                            format!(" ({:.2}seconds)", ms / 1000.0)
                        } else {
                            "".to_string()
                        }
                    );
                } else {
                    println!("Start computation...");
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(num_threads)
                        .build()
                        .unwrap();
                    let start = Instant::now();
                    let _res = pool.install(|| polynomial_multiply(&poly_a, &poly_b));
                    let duration = start.elapsed();
                    let ms = duration.as_secs_f64() * 1000.0;
                    println!(
                        "Parrallel Mission comoplete！execution time：{:.2}  milliseconds{}",
                        ms,
                        if ms > 1000.0 {
                            format!(" ({:.2} seconds)", ms / 1000.0)
                        } else {
                            "".to_string()
                        }
                    );
                }
                println!("----------------------------------------");
            }
        }
        _ => {
            println!("Invalid choice");
        }
    }
}
