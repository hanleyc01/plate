(define (fib n)
  (if (eq? n 0)
      0
      (if (eq? n 1)
          1
          (+ (fib (- n 1))
             (fib (- n 2))))))

(define (fibiter n a b)
  (if (eq? n 0)
      a
      (fibiter (- n 1) b (+ a b))))

(fib 10)
(fibiter 50 0 1)
