#ifndef HASHCOUNTER_H
#define HASHCOUNTER_H

#include <map>

class HashCounter
{
public:
    HashCounter() { map = std::map<int, int>(); }
    void increment(int key){
        if(map.find(key) == map.end()) {
            map[key] = 0;
        } else {
            map[key] = map[key] + 1;
        }
    }
    int operator [](int key){
        if(map.find(key) == map.end()) {
            return NULL;
        } else {
            return map[key];
        }
    }
private:
    std::map<int,int> map;
};

#endif // HASHCOUNTER_H
