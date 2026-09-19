(begin)

(begin
  (define width 10)
  (define height 20))

(begin
  (define origin 0)
  (begin
    (define unit 1)
    (define (step x) (+ x unit))))

(define (scale k)
  (begin
    (define w (* k width))
    (define h (* k height))
    (* w h)))

(begin 1 2 3)

(begin
  (define tmp (step origin))
  (begin
    (define more (step tmp)))
  (+ tmp more))
