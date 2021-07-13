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
     vector<ll> a(n);
     inarr(a,n);
     sort(a.begin(),a.end());
     map<ll,ll> m;
     for(int i=0;i<n;i++){
         m[a[i]]++;
     }
     map<ll,ll>::iterator itr;
     vector<ll> b;
     for(itr=m.begin();itr!=m.end();itr++){
         b.push_back(itr->second);
     }
    sort(b.begin(),b.end());
    ll s=0;
    for(int i=0;i<b.size()-1;i++){
        s+=b[i];
    }
    if(s<b[b.size()-1]){
        cout<<b[b.size()-1]-s<<endl;
    }
    else{
        if((s+b[b.size()-1])%2==0 ){
            cout<<0<<endl;
        }
        else{
            cout<<1<<endl;
        }
    }
    }   
}