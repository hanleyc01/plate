(define definex 1)

(define (iffy lambdax)
  (+ lambdax definex))

(define car?
  (lambda (x) (atom? x)))

(define nilly nil)

(define is?it? #t)

(define (begins x) x)

(iffy (begins definex))
