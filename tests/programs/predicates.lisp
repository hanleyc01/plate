(define (both? a b)
  (and a b))

(define (number? x)
  (and (atom? x) (int? x)))

(atom? nil)
(int? 42)
(int? #t)
(eq? #t #f)
(both? (int? 1) (atom? (cons 1 nil)))
(car (cons 1 2))
(cdr (cons 1 2))
