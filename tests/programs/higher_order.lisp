(define (compose f g)
  (lambda (x) (f (g x))))

(define (adder n)
  (lambda (x) (+ x n)))

(define twice
  (lambda (f x) (f (f x))))

(define (foldl f acc xs)
  (if (eq? xs nil)
      acc
      (foldl f (f acc (car xs)) (cdr xs))))

((adder 5) 10)
((compose (adder 1) (adder 2)) 0)
(twice (adder 3) 4)
((lambda (x y) (* x y)) 6 7)
(foldl (lambda (acc x) (+ acc x))
       0
       (cons 1 (cons 2 nil)))
