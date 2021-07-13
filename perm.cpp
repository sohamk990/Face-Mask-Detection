#include<iostream>
#include <string>
#include<bits/stdc++.h>
#include <fstream>
#define ll long long int
#define pb push_back
#define mp make_pair
#define M 1000000007
#define inarr(arr,n); for(ll i=0;i<n;i++) cin >> arr[i];
#define outarr(arr,n); for(ll i=0;i<n;i++) cout<<arr[i]<<" ";
using namespace std;

int main(){
    int t;
    cin>>t;
    while(t--){
        int n;
        cin>>n;
        vector<ll> q(n);
        inarr(q,n);
        vector<int> a(n+1);
        vector<int> p(n);
                vector<int> b(n+1);
        vector<int> r(n);
        a[0]=1;
        b[0]=1;
        p[0]=q[0];
        a[q[0]]=1;
        int x=1;
        for(int i=1;i<n;i++){
            if(q[i]==q[i-1]){
                if(a[x]==1){
                    while(a[x]==1){
                        x++;
                    }
                    p[i]=x;
                    a[x]=1;
                }
                else{
                    p[i]=x;
                    a[x]=1;
                    x++;
                }
                
            }
            else{
                p[i]=q[i];
                a[q[i]]=1;
            }
        }
r[0]=q[0];
        b[q[0]]=1;
        x=1;
        for(int i=1;i<n;i++){
            if(q[i]==q[i-1]){
                x=q[i];
                while(b[x]==1){
                    x--;
                }
                r[i]=x;
                b[x]=1;
            }
            else{
                r[i]=q[i];
                b[q[i]]=1;
            }
        }
        outarr(p,n)
        cout<<endl;
        outarr(r,n);
        cout<<endl;
    }
}