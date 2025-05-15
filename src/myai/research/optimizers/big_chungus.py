
# implementation based on https://github.com/hiroyuki-kasai/GDLibrary/blob/master/line_search/strong_wolfe_line_search.m
def strong_wolfe_backtracking(f, x0=0, c1=1e-4, c2=0.9, maxiter=3, a_m=20, a_p=0, a_x=1, r = 0.8, maxzoom=5):

    fx0, gx0 = f(x0)

    fxp = fx0
    # gxp = gx0
    i=1

    while True:
        xx = x0 + a_x
        fxx, gxx = f(xx)

        # fs = fxx
        # gs = gxx

        if (fxx > fx0 + c1*a_x*gx0) or ((i > 1) and (fxx >= fxp)):
            a_s = zoom(f,x0,a_p,a_x,fx0,gx0,c1=c1,c2=c2,maxiter=maxzoom)
            return a_s

        if abs(gxx) <= -c2*gx0:
            a_s = a_x
            return a_s

        if gxx >= 0:
            a_s = zoom(f,x0,a_x,a_p,fx0,gx0,c1=c1,c2=c2,maxiter=maxzoom)
            return a_s


        a_p = a_x
        fxp = fxx
        # gxp = gxx

        if i > maxiter:
            a_s = a_x
            return a_s


        a_x = a_x + (a_m-a_x)*r
        i = i+1




def zoom(f,x0,a_l,a_h,fx0,gx0, c1, c2, maxiter):
    i =0

    while True:
        # bisection
        a_x = 0.5*(a_l+a_h)
        a_s = a_x
        xx = x0 + a_x
        fxx, gxx = f(xx)

        # fs = fxx
        # gs = gxx

        # xl = x0 + a_l

        fxl, _ = f(xx)

        if ((fxx > fx0 + c1*a_x*gx0) or (fxx >= fxl)):
            a_h = a_x
        else:
            if abs(gxx) <= -c2*gx0:
                a_s = a_x
                return a_s

            if gxx*(a_h-a_l) >= 0:
                a_h = a_l

            a_l = a_x

        i = i+1
        if i > maxiter:
            a_s = a_x
            return a_s