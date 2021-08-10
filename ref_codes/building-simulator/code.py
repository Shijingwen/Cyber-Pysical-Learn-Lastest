import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import stats
import scipy.integrate
import scipy.optimize as optimize
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score


maxtimespace=200
maxitr=2
optmethod='SLSQP'

roomno=6
bigconf=1
smallconf=1
normalroom=roomno-bigconf-smallconf

nbc=1
nsc=1
nor1=1
nor2=1
nor3=1
nor4=1

wb=100
ws=20
wn=1

tmax = 10 # in hour u can assume

N = 20
permanT = 0.8

x0 = 67

initu=0
Tmax=80
Tmin=70

alpha=1
beta=0
gamma=0
phi=0

nmu, nsigma = 0, 0.1 # mean and standard deviation for noise
avg, std= 85 , 3 # mean and standard deviation for Texternal


# color for different rooms + 1 for Texternal
col=['red','purple','green','black','blue'] 

t = np.linspace(0.0, tmax, maxtimespace)
#print(t)

delt=tmax/maxtimespace
print("Delt= ",delt)





############# Generating Td ##################

td=[20 for i in range(roomno)]
td[0]=45


######### Tmob generation ###########


def sum_to_x(n, x):
    values = [0.0, x] + list(np.random.randint(low=0,high=x,size=n-1))
    values.sort()
    return [values[i+1] - values[i] for i in range(n)]

def generate_weighted_mobmap(bigconf,smallconf,normalroom,wb,ws,wn):
    map_weightrange={}
    sum=0
    j=0
    for i in range(0,bigconf):
        map_weightrange[i]=[sum+1,sum+wb]
        sum+=wb
        print("Room ",i,' Weightedrange ',map_weightrange[i])
    for i in range(bigconf,bigconf+smallconf):
        map_weightrange[i]=[sum+1,sum+ws]
        sum+=ws
        #print("Room ",i,' Weightedrange ',map_weightrange[i])
    for i in range(bigconf+smallconf,bigconf+smallconf+normalroom):
        map_weightrange[i]=[sum+1,sum+wn]
        sum+=wn
        #print("Room ",i,' Weightedrange ',map_weightrange[i])
    return map_weightrange
map_weightrange=generate_weighted_mobmap(bigconf,smallconf,normalroom,wb,ws,wn)

def assignpeople_per_hour():
    peopleinroom=[0 for i in range(roomno)]
    High=bigconf*wb + smallconf*ws + normalroom*wn
    for i in range(0,N):
        val=np.random.randint(low=1,high=High,size=1)
        #print(val)
        for r in map_weightrange.keys():
            rnge=map_weightrange[r]
            if val[0]>= rnge[0] and val[0]<=rnge[1]:
                peopleinroom[r]+=1
    
    return peopleinroom
#peopleinroom=assignpeople_per_hour()



def generate_tmob_list():
    
    mat=[[] for i in range(0,roomno)]
    #print(mat)
    print(len(mat))
    print(len(mat[0]))
    for i in range(0,tmax):
        mob=assignpeople_per_hour()
        #print(mob)
        for j in range(0,int(maxtimespace/tmax)):
            for k in range(0,roomno):
                #print("mob ",mob[k])
                
                mat[k].append(mob[k]*permanT)
                #print("mat ",mat)
                
                
                
            
        
    print(len(mat),len(mat[0]))
    return mat
Tmat=generate_tmob_list()




############### Generating Textenal ##################




bsig=0
ssig=0
o1sig=0
o2sig=0
o3sig=0
o4sig=0

def generate_roomcoupling():
    
    bc=[i for i in range(0,nbc)]
    sc=[i for i in range(nbc,nbc+nsc)]
    or1=[i for i in range(nbc+nsc,nbc+nsc+nor1)]
    or2=[i for i in range(nbc+nsc+nor1,nbc+nsc+nor1+nor2)]
    or3=[i for i in range(nbc+nsc+nor1+nor2,nbc+nsc+nor1+nor2+nor3)]
    or4=[i for i in range(nbc+nsc+nor1+nor2+nor3,nbc+nsc+nor1+nor2+nor3+nor4)]

    return [bc,sc,or1,or2,or3,or4]

room_coupling = generate_roomcoupling()
#print(room_coupling)


def generate_sig():
    sig=[]
    for i in range(0,len(room_coupling)):
        rooms=room_coupling[i]
        for j in range(0,len(rooms)):
            if i==0:
                sig.append(bsig)
            elif i==1:
                sig.append(ssig)
            elif i==2:
                sig.append(o1sig)
            elif i==3:
                sig.append(o2sig)
            elif i==4:
                sig.append(o3sig)
            elif i==5:
                sig.append(o4sig)
    return sig

tsig=generate_sig()
#print(tsig)
            
def generate_combination(lst):
    map_comb={}
    #print(lst)
    for i in range(0,len(lst)):
        for j in range(i,len(lst)):
            #print(lst[i],"  ",lst[j])
            map_comb[lst[i]]=lst[j]
    return map_comb
#lst=[1,2,3,4,5]
#generate_combination(lst)
            
def generate_covariancematrix():
    cov=np.zeros((roomno,roomno), dtype=int)
    for rooms in room_coupling:
        comb=generate_combination(rooms)
        for key in comb.keys():
            cov[key][comb[key]]=1
            cov[comb[key]][key]=1
    return cov

cov=generate_covariancematrix()
#print(cov[0])

def generate_multivariate():

    x = np.random.multivariate_normal(tsig, cov, len(t))
    #print(x[:,51])
    return x
multivariate=generate_multivariate()

def markov_process(prevtex,room,time):
    return prevtex+((1/5)*multivariate[time][room])
    

def generate_basetemp(room):
    base=0
    for i in range(0,len(room_coupling)):
        rooms=room_coupling[i]
        if room in rooms:
            if i==0:
                base=82
            elif i==1:
                base=85
            elif i==2:
                base=80
            else:
                base=81
    return base


def generate_tex_list():
    avg, std = 85, 3 # mean and standard deviation
    Textern=[]
    for i in range(0,tmax):
        tex=np.random.normal(avg, std, 1)
        for j in range(0,int(maxtimespace/tmax)):
            Textern.append(tex)
    return Textern

#Textern=generate_tex_list()

            



############### Generating AC controller input ################
        


def u(t,Tnow,uprev):
    if Tnow>=Tmax:
        return 1
    elif Tnow<=Tmin:
        return 0
    else:
        return uprev
    




  
def RunODE():
    T=[]
    Tex=[]
    U=[]
    for i in range(0,roomno):
        
        """
        args = (Tex,td,tm,i,u)
        y = sp.integrate.odeint(func, x0, t, args)
        """
        y=[]
        sig=[]
        ycurrent=x0
        del_t=tmax/maxtimespace
        uprev=initu
        j=0
        #Textern=generate_tex_list()
        Textern=[]
        prevtex=generate_basetemp(i)
        for time in t:
            uprev=u(time,ycurrent,uprev)
            tmob=Tmat[i][j]

            #generate textern base temp
            
            tex=markov_process(prevtex,i,j)
            #print("Tex: ",tex)
            prevtex=tex
            
            ynext=((-1*alpha*ycurrent + beta*tex + gamma*tmob - phi*td[i]*uprev)*del_t)+ycurrent

            y.append(ynext)
            sig.append(uprev)
            Textern.append(tex)
            
            ycurrent=ynext

            j+=1

        
        xfont = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}
        yfont = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 10}

        fig = plt.figure()
        ax = fig.add_subplot(411)
        h1, = ax.plot(t, y,c=col[i%3])
        h2, = ax.plot(t, Textern, c=col[len(col)-1])
        
        
        ax.legend([h1, h2], ["Actual Room Temp", "Texternal"])
        #ax.set_xlabel("time")
        plt.ylabel("Temparature ",**yfont)
        ax.grid()
        #plt.show()

        ax = fig.add_subplot(412)
        """
        leg_handle=[]
        leg_label=[]
        
        for x in range(0,roomno):
            h3, = ax.plot(t, Tmat[x],c=col[x])
            leg_handle.append(h3)
            leg_label.append("Room "+str(x+1))
        """  
        h4, = ax.plot(t, Tmat[i],c=col[i%3])
        ax.legend([h4], ["Room "+str(i+1)])
        #ax.set_xlabel("time")
        plt.ylabel("Tmob  =~ # of people",**yfont)

        
        ax = fig.add_subplot(413)
        h5, = ax.plot(t, sig,c='y')
        #ax.set_xlabel("time")
        #plt.xlabel("Time",**xfont)
        plt.ylabel("AC Controller Input",**yfont)
        #plt.show()

        #print("len y:", len(y))

        
        s = np.random.normal(nmu, nsigma, maxtimespace)
        #print("len s: ",len(s))

        Tfinal=[]
        for j in range(0,len(s)):
            Tfinal.append(s[j]+y[j])
        

        ax = fig.add_subplot(414)
        plt.plot(t,Tfinal,c=col[i%3])
        plt.xlabel("Time",**xfont)
        plt.ylabel("Noisy Temp- Room "+str(i+1),**yfont)

        directory="figs/Room "+str(i+1)
        #plt.savefig(directory)
        #plt.show()
        plt.close()

        
        
        """
        lines = [line.rstrip('\n') for line in open('inputu.txt')]
        print("Read lines: \n\n\n\n\n\n\n\n\n\n")
        print(lines)
        """
        #print("Fig ",i)
        Tfinal.append(Tfinal[len(Tfinal)-1])
        T.append(Tfinal)
        Tex.append(Textern)
        U.append(sig)
    return T,Tex,U
# T length = roomno * maxtimespace+1
# Tex length = roomno * maxtimespace
# U length = roomno * maxtimespace
# Tmat length = roomno * maxtimespace
# td length = roomno

T,Tex,U = RunODE()


Tmobsim=Tmat













#############################    OPTIMIZE #################


        #print("len s: ",len(s))
Texfinal=[]
for i in range(0,roomno):
    temp=[]
    s = np.random.normal(nmu, nsigma, maxtimespace)
    for j in range(0,len(s)):
        temp.append(s[j]+Tex[i][j])
    Texfinal.append(temp)
Tex=Texfinal

"""
tmat=[]
for i in range(0,roomno):
    temp=[]
    for j in range(0,maxtimespace):
        if i==0:
            temp.append(N)
        else:
            temp.append(0)
    tmat.append(temp)

Tmob = np.array(tmat)
"""
tmat= generate_tmob_list()
Tmob = np.array(Tmat) #### original simulation is Tmat

def to1d(w,m,n):
    return np.hstack([w.flatten()])

def to2d(vec,m,n):
    return vec[:m*n].reshape(m,n)

tm1d=to1d(Tmob,roomno,maxtimespace)
print(len(tm1d))

#############################  INITIALIZATON #############
a = 0
b = 0
c = 0

init=[a,b,c]
#init=[a]
for i in range(0,len(tm1d)):
    init.append(tm1d[i])

#init=[a,b,c]
print(len(init))



############ Constraints ###########


def constraint(x,col):
    x=x[3:]
    x=to2d(x,roomno,maxtimespace)
    x=x[:,col]
    return x.sum() - N*permanT

def set_constraint():
    mobcons=[]
    for i in range(maxtimespace):
        con={'type': 'eq', 'fun': constraint,'args':(i,)}
        mobcons.append(con)
    return mobcons

mobcons=set_constraint()


############ Bounds ################



def set_bounds():
    b=[]
    b.append((0.95,1.05))
    b.append((0.95,1.05))
    b.append((0.95,1.05))
    for i in range(3,(roomno*maxtimespace)+3):
        b.append((0,None))
    return b
bnds=set_bounds()
"""
b=[]
b.append((0,None))
b.append((0,None))
b.append((0,None))
bnds=b

"""


def func(param):
    a=param[0]
    
    b=param[1]
    
    c=param[2]
    
    tm=param[3:]
    tm=to2d(tm,roomno,maxtimespace)
    
    
    sum=0
    
    for i in range(0,roomno):
        for j in range(0,maxtimespace):
            #sum  += (((T[i][j+1]- T[i][j])/delt) + a*T[i][j] - b*Tex[i][j] - c * td[i] * U[i][j] - tm[i][j])**2
            sum+=(((T[i][j+1]- T[i][j])/delt) + a*T[i][j] - b*Tex[i][j] + c * td[i] * U[i][j] - tm[i][j])**2
    #print("Sum : ",sum)
    return sum

def optimizeAndshowresult():
    print("Calculating optimization ... ")
    res = optimize.minimize(func, init, method=optmethod, bounds=bnds,constraints=mobcons,
                            options={'maxiter':maxitr,'disp': False})

    print("Processing Results ... ")

    res=res.x
    a=res[0]
    
    b=res[1]
    
    c=res[2]
    

    mobt=res[3:]
    #print(mobt)
    print(len(mobt))

    print("a = ",a)
    print("b = ",b)
    print("c = ",c)

    mobt=to2d(mobt,roomno,maxtimespace)
    print(sum(mobt[:,0]))

    return a,b,c,mobt
    """
    return a,b,c
    """

a,b,c,mobt = optimizeAndshowresult()
#a,b,c= optimizeAndshowresult()
#print("final a: ",a," final b: ",b," final c: ",c)
def plot_inference():
    for i in range(roomno):
        diff=list(np.array(Tmobsim[i]) - np.array(mobt[i]))
        plt.plot(t,diff,'r')
        plt.xlabel("Time")
        plt.ylabel("Tmob difference between Original vs Infered- Room "+str(i+1))
        directory="mobdifffig/room "+str(i+1)
        plt.savefig(directory)
        plt.show()
        plt.close()

plot_inference()


for i in range(roomno):
    rs=r2_score(Tmobsim[i], np.array(mobt[i]))
    mae=median_absolute_error(Tmobsim[i], np.array(mobt[i]))
    print("Room ",i," Rscore: ",rs,' MAE: ',mae)

    


        







